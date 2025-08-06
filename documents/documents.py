from typing import TypedDict
from bson import ObjectId

TXT_VEC_FIELD = 'text_vector'
IMG_VEC_FIELD = 'image_vector'

class FlatDoc(TypedDict):
    _id: ObjectId
    img: str
    description: str | None
