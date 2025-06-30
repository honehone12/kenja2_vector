import asyncio
import os
from typing import final
from urllib.parse import urlparse
from dotenv import load_dotenv
from bson import ObjectId
from pymongo import DeleteOne, UpdateOne
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.cursor import AsyncCursor
from interfaces.vgen import ImageVGen, TextVGen
from db import mongo
from documents.documents import*
from models.embed_text_v2 import EmbedTextV2
from models.siglip import Siglip

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

def process_image(
    img_root: str,
    url: str, 
    id: ObjectId,
    img_gen: ImageVGen 
):
    if len(url) == 0:
        raise ValueError('empty url')

    path = img_root + urlparse(url).path.removesuffix('/')
    if not os.path.exists(path):
        print(f'image not found {path}')
        d = DeleteOne(
            filter={'_id': id}
        )
        return d, False

    v = img_gen.gen_image_vector(path)
    b = mongo.compress_bin(v)
    u = UpdateOne(
        filter={'_id': id},
        update={
            '$set': {IMG_VEC_FIELD: b},
            '$unset': {'img': ''}
        }
    )
    return u, True

def process_text(
    field: str,
    text: str, 
    id: ObjectId,
    txt_gen: TextVGen 
):
    if len(text) == 0:
        raise ValueError('empty text')

    v = txt_gen.gen_text_vector(text)
    b = mongo.compress_bin(v)
    u = UpdateOne(
        filter={'_id': id},
        update={'$set': {field: b}}
    )
    return u

async def gen_vectors(
    args: Args,
    img_gen: ImageVGen,  
    txt_gen: TextVGen
):
    db = mongo.db('DATABASE')
    cl: AsyncCollection[FlatDoc] = mongo.collection(db, 'COLLECTION')
    stream: AsyncCursor[FlatDoc] = cl.find({})

    it = 0
    batch = []
    async for doc in stream:
        id = doc['_id']

        if doc.get(IMG_VEC_FIELD) is None:
            op, ok = process_image(
                args.img_root, 
                doc['img'], 
                id, 
                img_gen
            )
            batch.append(op)
            if not ok:
                continue
            
        if doc.get(TXT_VEC_FIELD) is None:
            op = process_text(
                TXT_VEC_FIELD, 
                doc['description'], 
                id,
                txt_gen
            )
            batch.append(op)

        if doc.get(STF_VEC_FIELD) is None:
            op = process_text(
                STF_VEC_FIELD, 
                doc['staff'], 
                id,
                txt_gen
            )
            batch.append(op)

        if len(batch) >= args.batch_size:
            res = await cl.bulk_write(batch)
            print(f'{res.modified_count} updated')
            batch.clear()

        it += 1
        print(f'iteration {it} done')
        if it >= args.iteration:
            print('iteration limit')
            break

    if len(batch) > 0:
        res = await cl.bulk_write(batch)
        print(f'{res.modified_count} updated')

    print('done')

if __name__ == '__main__':
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

        img_gen = Siglip()
        txt_gen = EmbedTextV2()
        args = Args(iteration, batch_size, img_root)
        mongo.connect()
        asyncio.run(gen_vectors(args, img_gen, txt_gen))
    except Exception as e:
        print(e)
