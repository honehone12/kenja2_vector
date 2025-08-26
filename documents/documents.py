from typing import TypedDict
from bson import ObjectId

class FlatDoc(TypedDict):
    _id: ObjectId
    img: str
