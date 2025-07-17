import random
from torch import Tensor
from PIL.Image import Image
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
        self.img_processor = self._load_img_process(img_process)
        self._data = self._load_data_hf(data_name, split)


    
    def _preprocess_data(self, data: DatasetDict) -> list[tuple[tuple[str], Image]]:
        """
        Preprocesses the data by extracting questions, answers, and images.
        Returns a list of tuples containing question-answer pairs and the image.

        Args:
            data (DatasetDict): Dataset downloaded from Hugging Face
        Returns:
            list[tuple[tupe[str], Image]]: List of tuples with question-answer pairs and image
        """
        processed_data = []
        count = 1
        for i, item in enumerate(data):
            if count % 100 == 0:
                break
            
            txt, label = self._process_txt(item["dialog"])
            if not txt or not label:
                continue

            image = self._process_img(item["image"])   
            processed_data.append((image, txt, label))
            count += 1
        return processed_data


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
            data = load_dataset(url, split=split, num_proc=1, keep_in_memory=False)
            data = data.remove_columns(
                [
                    "caption", 
                    "image_path", 
                    "global_image_id", 
                    "anns_id"
                ]
            )
            preprocess_data = self._preprocess_data(data)
        except HuggingFaceDataNotFound:
            log.error(f"Dataset {url} not found")
            raise
        return preprocess_data


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
        qa_pairs, last_ans = "", ""
        for i, qa in enumerate(txt):
            # Ensure that both question and answer are non-empty
            if not qa[0] or not qa[1]:
                continue
            
            #TODO: Add special case where last qa doesnt pass above condition
            #TODO: Add test cases to validate data preprocessing
            if i+1 >= len(txt):
                qa_pairs += f"[Q] {qa[0]} [/Q] <EOS>"
            else:
                qa_pairs += f"[Q] {qa[0]} [/Q] {qa[1]} "
            last_ans = qa[1]
            
        return qa_pairs, last_ans
    

    def __len__(self) -> int:
        """Returns the length of batched dataset"""
        return len(self._data)


    def __getitem__(self, idx: int) -> dict[str: Tensor]:
        """Preprocesses each datapoint and returns image, text, and masking tensors"""
        img, txt, label = self._data[idx]
        return {
            "img": img, 
            "txt": txt,
            "label": label
        }