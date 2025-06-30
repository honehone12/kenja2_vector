from abc import ABC, abstractmethod
import torch

class TextVGen(ABC):
    @abstractmethod
    def gen_text_vector(self, text: str) -> torch.Tensor:
        pass

class ImageVGen(ABC):
    @abstractmethod
    def gen_image_vector(self, path: str) -> torch.Tensor:
        pass
