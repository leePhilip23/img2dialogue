import torch
from torch import Tensor
from typing import Any
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BlipImageProcessorFast,
    CLIPImageProcessorFast,
)
from utils import log
from __future__ import annotations


class HuggingFaceDataNotFound(Exception):
    """Hugging Face Dataset not found"""
    pass


class ImgProcessorNotFound(Exception):
    """Hugging Face Image Processor not found"""
    pass


class TxtTokenizerNotFound(Exception):
    """Hugging Face Text Tokenizer not found"""
    pass


class CustomImageDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_name: str = "HuggingFaceM4/VisDial",
        img_process: str = "Salesforce/blip-image-captioning-base",
        txt_process: str = "Qwen/Qwen3-0.6B",
    ):
        self._data = self._load_data_hf(data_name, split)
        self.img_processor = self._load_img_process(img_process)
        self.txt_tokenizer = self._load_txt_token(txt_process)


    def _load_data_hf(self, url: str, split: str) -> CustomImageDataset:
        """
        Loads the data and returns it

        Args:
            url (str): Directory of Hugging Face dataset
        
        Returns: 
            CustomImageDataset: Dataset downloaded from Hugging Face

        Raises:
            HFDataNotFound
        """
        try:
            data = load_dataset("HuggingFaceM4/VisDial")
            data = data.remove_columns(
                ["caption", "image_path", "global_image_id", "anns_id"]
            )
        except HuggingFaceDataNotFound:
            log.exception(f"Dataset {url} not found")
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
            if "blip" in img_model_name.lower():
                processor = BlipImageProcessorFast.from_pretrained(img_model_name)
            elif "clip" in img_model_name.lower():
                processor = CLIPImageProcessorFast.from_pretrained(img_model_name)
        except ImgProcessorNotFound:
            log.exception(
                f"Processor not found for: {img_model_name} in Hugging Face Directory"
            )
        return processor


    def _load_txt_token(self, txt_token_name: str) -> AutoTokenizer:
        """
        Loads the text tokenizer and returns it

        Args:
            url (str): Directory of Hugging Face Tokenizer
        
        Returns: 
            AutoTokenizer: Tokenizer downloaded from Hugging Face

        Raises:
            TxtTokenizerNotFound
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                txt_token_name, padding_side="left", bos_token="<BOS>", eos_token="<EOS>"
            )
        except TxtTokenizerNotFound:
            log.exception(
                f"Tokenizer not found for: {txt_token_name} in Hugging Face Directory"
            )
        return tokenizer


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
        dialogue = " ".join(f"[Q] {q} [/Q] [A] {a} [/A]" for q, a in txt)
        dialogue = f"<BOS> {dialogue} <EOS>"
        inputs_masks = self.txt_tokenizer(
            dialogue, padding=True, return_tensors="pt"
        )
        return inputs_masks.input_ids[0], inputs_masks.attention_mask[0]


    def __len__(self) -> int:
        """Returns the length of batched dataset"""
        return len(self._data)


    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Preprocesses each datapoint and returns image, text, and masking tensors"""
        txt, img = self._data[idx]
        img_input = self._process_img(img)
        txt_input, txt_mask = self._process_txt(txt)
        return img_input, txt_input, txt_mask
