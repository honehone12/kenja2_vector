import asyncio
import os
from typing import final
from dotenv import load_dotenv
from pymongo import UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.cursor import AsyncCursor
from db import mongo
from documents.documents import*
from interfaces.vgen import ImageVGen
from models.siglip2 import Siglip2
from logger.logger import init_logger, log

@final
class Envs:
    def __init__(self):
        img_root = os.getenv('IMG_ROOT')
        if img_root is None:
            raise ValueError('env for image root is not set')

        batch_size = 100
        batchenv = os.getenv('BATCH_SIZE')
        if batchenv is not None:
            batch_size = int(batchenv)

        self.batch_size = batch_size
        self.img_root = img_root

async def gen_vectors(
    envs: Envs,
    mongo_client: mongo.MongoClient,
    img_gen: ImageVGen
):
    l = log()
    cl: AsyncCollection[FlatDoc] = mongo_client.collection()
    stream: AsyncCursor[FlatDoc] = cl.find({})
    l.info('converting stream to list...')
    list = await stream.to_list()

    it = 0
    batch = []
    for doc in list:
        id = doc['_id']

        if doc.get('image_vector') is None:
            img_name = doc['img']
            if len(img_name) == 0:
                raise ValueError('empty url')

            path = f'{envs.img_root}/{img_name}'
            if not os.path.exists(path):
                raise ValueError(f'image not found {path}')

            v = img_gen.gen_image_vector(path)
            b = mongo.compress_bin_i8(v)
            u = UpdateOne(
                filter={'_id': id},
                update={'$set': {'image_vector': b}}
            )
            batch.append(u)

        if len(batch) >= envs.batch_size:
            res = await cl.bulk_write(batch)
            l.info(f'{res.modified_count} updated')
            batch.clear()

        it += 1
        l.info(f'iteration {it} done')

    if len(batch) > 0:
        res = await cl.bulk_write(batch)
        l.info(f'{res.modified_count} updated')

    l.info('done')

if __name__ == '__main__':
    init_logger(__name__)

    if not load_dotenv():
        raise RuntimeError('failed to initialize dotenv')

    envs = Envs()
    img_gen = Siglip2()
    mongo_client = mongo.MongoClient()
    asyncio.run(gen_vectors(envs, mongo_client, img_gen))
