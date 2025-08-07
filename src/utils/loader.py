from torch import Tensor
from PIL.Image import Image
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import (
    BlipImageProcessorFast,
    CLIPImageProcessorFast,
)
from src import log
from .exceptions import (
    HuggingFaceDataNotFound,
    ImgProcessorNotSupported,
    ImgProcessorNotFound
)


class CustomDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_name: str = "lmms-lab/LLaVA-NeXT-Data",
        img_process: str = "Salesforce/blip-image-captioning-base"
    ):
        self.img_processor = self._load_img_process(img_process)
        self._data = self._load_data_hf(data_name, split)


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
            data = load_dataset(url, split=split, num_proc=1, keep_in_memory=True, seed=None)
            data = data.remove_columns(["id", "data_source"])
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
            else:
                log.error(f"Processor: {img_model_name} is not a valid choice")
                raise ImgProcessorNotSupported("Processor type is not among the supported choices")
        except ImgProcessorNotFound:
            log.error(f"Image Processor: {img_model_name} is not found in HF repository")
            raise


    def _process_img(self, img: Image) -> Tensor:
        """Returns preprocessed and normalized image from raw image"""
        return self.img_processor(img).pixel_values[0]


    def _process_txt(self, txt: dict[str, str]) -> tuple[str, str]:
        """
        Preprocesses by adding special tokens to each question and answer.
        Then it tokenizes the raw text.

        Args:
            txt (list[list[str]]): Each matrix row is a question and answer
        Returns:
            tuple[str, str]: Preprocessed text and attention masks to
                                   prevent gradient calculations for padding
        """
        qa_pairs = "<SOS> ", 
        for i in range(len(txt)-1):
            if txt[i]['from'] == "human":
                qa_pairs += f"[Q] {txt[i]['value']} [/Q] "
            else:
                qa_pairs += f"{txt[i]['value']} "
        return qa_pairs + "<EOS>", txt[-1]['value']
    

    def _preprocess_data(self, data: DatasetDict) -> list[tuple[Image, str, str]]:
        """
        Preprocesses the data by extracting questions, answers, and images.
        Returns a list of tuples containing question-answer pairs and the image.

        Args:
            data (DatasetDict): Dataset downloaded from Hugging Face
        Returns:
            list[tuple[tupe[str], Image]]: List of tuples with question-answer pairs and image
        """
        processed_data = []
        for item in data:
            image = self._process_img(item["image"])  
            txt, label = self._process_txt(item["conversations"]) 
            processed_data.append((image, txt, label))
        return processed_data
    

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