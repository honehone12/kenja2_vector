from typing import TypedDict
from bson import ObjectId


class ImageDoc(TypedDict):
    _id: ObjectId
    img: str
