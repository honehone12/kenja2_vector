from typing import TypedDict
from bson import ObjectId

RATING_ALL_AGES = 1
RATING_HENTAI = 2

TXT_VEC_FIELD = 'text_vector'
IMG_VEC_FIELD = 'image_vector'

class FlatDoc(TypedDict):
    _id: ObjectId
    rating: int
    img: str
    description: str
