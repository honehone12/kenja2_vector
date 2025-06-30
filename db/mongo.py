import os
from typing import final
from pymongo import AsyncMongoClient
from bson.binary import Binary, BinaryVectorDtype

@final
class MongoClient:
    def __init__(self):
        uri = os.getenv('MONGO_URI')
        if uri is None:
            raise ValueError('env for mongo uri not set')
        
        db = os.getenv('MONGO_DB')
        if db is None:
            raise ValueError(f'env for mongo db not set')
        
        cl = os.getenv("MONGO_CL")
        if cl is None:
            raise ValueError(f'env for mongo collection is not set')    
        
        self._client = AsyncMongoClient(uri)
        self._db = self._client[db]
        self._cl = self._db[cl]

    def database(self):
        return self._db

    def collection(self):
        return self._cl

def compress_bin(vector):
    return Binary.from_vector(vector, dtype=BinaryVectorDtype.FLOAT32)
