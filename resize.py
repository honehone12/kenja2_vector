import os
from PIL import Image
from dotenv import load_dotenv
from logger.logger import init_logger, log

__RESIZE_LIST = [
    'angry.png',
    'happy.png',
    'hope.png',
    'sad.png',
    'safe.png',
    'strong.png',
    'surprise.png',
    'unreal.png',
]

def resize_imgs(img_root: str):
    size = (256, 256)
    for file_name in __RESIZE_LIST:
        src = f'{img_root}/raw/{file_name}'
        dst = f'imgs/{file_name}'
        with Image.open(src) as img:
            img = img.resize(size)
            img.save(dst)

if __name__ == "__main__":
    init_logger(__name__)

    try:
        if not load_dotenv():
            raise RuntimeError('failed to initialize dotenv')
        
        img_root = os.getenv('RESIZE_IMG_ROOT')
        if img_root is None:
            raise ValueError('env for image root is not set')
        
        resize_imgs(img_root)

    except Exception as e:
        log().error(e)
