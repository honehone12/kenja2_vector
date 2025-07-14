import os
from typing import override, final
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from PIL import Image
from interfaces.vgen import ImageVGen

@final
class Siglip2(ImageVGen):
    def __init__(self):
        model_name = os.getenv('IMG_EMBED_MODEL')
        if model_name is None:
            raise ValueError('env for siglip model is not set')

        self._device = torch.device('cuda')
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval().to(self._device)
        self._processor = AutoProcessor.from_pretrained(
            model_name, 
            use_fast=True
        )

    @override
    def gen_image_vector(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        
        with torch.no_grad():
            params = self._processor(
                images=[img],
                return_tensors='pt'
            ).to(self._device)
            raw = self._model.get_image_features(**params)
            return F.normalize(raw, p=2.0, dim=-1).squeeze(0)
