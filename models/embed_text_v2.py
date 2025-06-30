import os
from typing import override, final
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from interfaces.vgen import TextVGen

@final
class EmbedTextV2(TextVGen):
    _PRONPT = 'passage'
    
    def __init__(self):
        model_name = os.getenv('TXT_EMBED_MODEL')
        if model_name is None:
            raise ValueError('env for text embed model is not set')

        self._model = SentenceTransformer(
            model_name,
            device='cuda',
            trust_remote_code=True
        )

    @override
    def gen_text_vector(self, text: str) -> torch.Tensor:
        raw = self._model.encode(
            text,
            prompt_name=EmbedTextV2._PRONPT,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        return F.normalize(raw, p=2.0, dim=-1)
