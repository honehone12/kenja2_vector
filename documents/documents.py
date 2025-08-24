from typing import TypedDict
from bson import ObjectId

IMG_VEC_FIELD = 'image_vector'

class FlatDoc(TypedDict):
    _id: ObjectId
    img: str
