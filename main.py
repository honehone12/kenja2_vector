import asyncio
import os
from typing import final
from dotenv import load_dotenv
from pymongo import UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.cursor import AsyncCursor
from interfaces.vgen import ImageVGen, TextVGen
from db import mongo
from documents.documents import*
from logger.logger import init_logger, log
from models.embed_text_v2 import EmbedTextV2
from models.siglip2 import Siglip2

@final
class Args:
    def __init__(
        self,
        iteration: int,
        batch_size: int,
        img_root: str 
    ):
        self.iteration = iteration
        self.batch_size = batch_size
        self.img_root = img_root

async def gen_vectors(
    args: Args,
    mongo_client: mongo.MongoClient,
    img_gen: ImageVGen,  
    txt_gen: TextVGen
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

        if doc.get(IMG_VEC_FIELD) is None:
            img_name = doc['img']
            if len(img_name) == 0:
                raise ValueError('empty url')

            path = f'{img_root}/{img_name}'
            if not os.path.exists(path):
                raise ValueError(f'image not found {path}')

            v = img_gen.gen_image_vector(path)
            b = mongo.compress_bin(v)
            u = UpdateOne(
                filter={'_id': id},
                update={'$set': {IMG_VEC_FIELD: b}}
            )
            batch.append(u)
            
        if doc.get(TXT_VEC_FIELD) is None:
            text = doc['description']
            if text is None:
                l.info('skipping null text')
            else:
                if len(text) == 0:
                    raise ValueError('empty text')

                v = txt_gen.gen_text_vector(text)
                b = mongo.compress_bin(v)
                u = UpdateOne(
                    filter={'_id': id},
                    update={'$set': {TXT_VEC_FIELD: b}}
                )
                batch.append(u)

        if len(batch) >= args.batch_size:
            res = await cl.bulk_write(batch)
            l.info(f'{res.modified_count} updated')
            batch.clear()

        it += 1
        l.info(f'iteration {it} done')
        if it >= args.iteration:
            l.info('iteration limit')
            break

    if len(batch) > 0:
        res = await cl.bulk_write(batch)
        l.info(f'{res.modified_count} updated')

    l.info('done')

if __name__ == '__main__':
    init_logger(__name__)

    try:
        if not load_dotenv():
            raise RuntimeError('failed to initialize dotenv')

        itenv = os.getenv('ITERATION')
        iteration = 100
        if itenv is not None:
            iteration = int(itenv)

        batchenv = os.getenv('BATCH_SIZE')
        batch_size = 100
        if batchenv is not None:
            batch_size = int(batchenv)

        img_root = os.getenv('IMG_ROOT')
        if img_root is None:
            raise ValueError('env for image root is not set')

        img_gen = Siglip2()
        txt_gen = EmbedTextV2()
        args = Args(iteration, batch_size, img_root)
        mongo_client = mongo.MongoClient()
        asyncio.run(gen_vectors(args, mongo_client, img_gen, txt_gen))
    except Exception as e:
        log().error(e)
