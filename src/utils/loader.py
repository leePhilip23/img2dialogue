import random
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import (
    BlipImageProcessorFast,
    CLIPImageProcessorFast,
)
from .exceptions import (
    HuggingFaceDataNotFound,
    ImgProcessorNotSupported,
    ImgProcessorNotFound
)
from utils import log


class CustomDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_name: str = "HuggingFaceM4/VisDial",
        img_process: str = "Salesforce/blip-image-captioning-base"
    ):
        self._data = self._load_data_hf(data_name, split)
        self.img_processor = self._load_img_process(img_process)


    def _load_data_hf(self, url: str, split: str) -> DatasetDict:
        """
        Loads the data and returns it

        Args:
            url (str): Directory of Hugging Face dataset
        
        Returns: 
            DatasetDict: Dataset downloaded from Hugging Face

        Raises:
            HFDataNotFound
        """
        try:
            data = load_dataset(url)
            data = data.remove_columns(
                ["caption", "image_path", "global_image_id", "anns_id"]
            )
        except HuggingFaceDataNotFound:
            log.error(f"Dataset {url} not found")
            raise
        return data[split]


    def _load_img_process(self, img_model_name: str) -> BlipImageProcessorFast | CLIPImageProcessorFast:
        """
        Loads the image processor and returns it

        Args:
            url (str): Directory of Hugging Face image processor
        
        Returns: 
            BlipImageProcessorFast | CLIPImageProcessorFast: Image Processor downloaded from Hugging Face

        Raises:
            ImgProcessorNotFound
        """
        try:
            if 'blip' in img_model_name.lower():
                return BlipImageProcessorFast.from_pretrained(img_model_name)
            elif 'clip' in img_model_name.lower():
                return CLIPImageProcessorFast.from_pretrained(img_model_name)
            else:
                log.error(f"Processor: {img_model_name} is not a valid choice")
                raise ImgProcessorNotSupported("Processor type is not among the supported choices")
        except ImgProcessorNotFound:
            log.error(f"Image Processor: {img_model_name} is not found in HF repository")
            raise


    def _process_img(self, img: Image) -> Tensor:
        """Returns preprocessed and normalized image from raw image"""
        return self.img_processor(img).pixel_values[0]


    def _process_txt(self, txt: list[list[str]]) -> tuple[Tensor, Tensor]:
        """
        Preprocesses by adding special tokens to each question and answer.
        Then it tokenizes the raw text.

        Args:
            txt (list[list[str]]): Each matrix row is a question and answer

        Returns:
            tuple[Tensor, Tensor]: Preprocessed text and attention masks to
                                   prevent gradient calculations for padding
        """
        r_index = random.randint(0, len(txt)-1)
        return f"[Q] {txt[r_index][0]} [/Q] {txt[r_index][1]} <EOS>"
    

    def __len__(self) -> int:
        """Returns the length of batched dataset"""
        return len(self._data)


    def __getitem__(self, idx: int) -> dict[str: Tensor]:
        """Preprocesses each datapoint and returns image, text, and masking tensors"""
        txt, img = self._data[idx]
        img_input = self._process_img(img)
        txt_input =  self._process_txt(txt)
        return {
            "img": img_input, 
            "txt": txt_input
        }