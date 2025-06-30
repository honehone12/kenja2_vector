import os
from typing import Any
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from bson.binary import Binary, BinaryVectorDtype

__client = None

def connect():
    global __client

    uri = os.getenv('MONGO_URI')
    if uri is None:
        raise ValueError('env for MONGO_URI not set')
    __client = AsyncMongoClient(uri)

def db(key: str):
    if __client is None:
        raise ValueError('client is not initialized')

    db = os.getenv('DATABASE')
    if db is None:
        raise ValueError(f'env for {key} not set')
    return __client[db]

def collection(db: AsyncDatabase[Any], key: str):
    collection = os.getenv(key)
    if collection is None:
        raise ValueError(f'env for {key} is not set')
    return db[collection]

def compress_bin(vector):
    return Binary.from_vector(vector, dtype=BinaryVectorDtype.FLOAT32)
