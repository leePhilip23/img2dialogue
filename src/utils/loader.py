import torch
from typing import Any
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BlipImageProcessorFast, CLIPImageProcessorFast
from datasets import load_dataset, DatasetDict
from utils import log
from __future__ import annotations


class HFDataNotFound(Exception):
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
        txt_process: str = "Qwen/Qwen3-0.6B"
    ):
        self._data = self._load_data_hf(data_name, split)
        self.img_processor = self._load_img_token(img_process)
        self.txt_tokenizer = self._load_txt_token(txt_process)
        

    def _load_data_hf(self, url: str, split: str) -> CustomImageDataset:
        try:
            data = load_dataset("HuggingFaceM4/VisDial")
            data = data.remove_columns(['caption', 'image_path', 'global_image_id', 'anns_id'])
        except HFDataNotFound:
            log.exception(f"Dataset {url} not found")
        return data[split]
    

    def _load_img_token(self, url: str) -> BlipImageProcessorFast | CLIPImageProcessorFast:
        try:
            if "blip" in url.lower():
                processor = BlipImageProcessorFast.from_pretrained(url)
            elif "clip" in url.lower():
                processor =  CLIPImageProcessorFast.from_pretrained(url)
        except ImgProcessorNotFound:
            log.exception(f"Processor not found for: {url} in Hugging Face Directory")
        return processor
    

    def _load_txt_token(self, url: str) -> AutoTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                url, 
                padding_side='left', 
                bos_token='<BOS>', 
                eos_token='<EOS>'
            )
        except TxtTokenizerNotFound:
            log.exception(f"Tokenizer not found for: {url} in Hugging Face Directory")
        return tokenizer


    def _process_img(self, img: Image) -> torch.tensor:
        return self.img_processor(img).pixel_values[0]


    def _process_txt(self, txt: list[list[str]]) -> tuple[list[int], list[int]]:
        dialogue = ' '.join(f"[Q] {q} [/Q] [A] {a} [/A]" for q, a in txt)
        dialogue = f"<BOS> {dialogue} <EOS>"
        inputs_masks =  self.txt_tokenizer(dialogue, padding=True, return_tensors="pt")
        return inputs_masks.input_ids[0], inputs_masks.attention_mask[0]
    

    def __len__(self):
        return len(self._data)
    
    
    def __getitem__(self, idx):
        txt, img = self._data[idx]
        img_input = self._process_img(img)
        txt_input, txt_mask = self._process_txt(txt)
        return img_input, txt_input, txt_mask