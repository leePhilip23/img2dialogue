import torch
import torch.nn.functional as F
from torch import Tensor
from PIL.Image import Image
from datasets import DatasetDict
from transformers import CLIPProcessor, CLIPModel
from conftest import sampled_data

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def _process_img(img: Image) -> Tensor:
    """Returns preprocessed and normalized image from raw image"""
    return processor(img).pixel_values[0]


def _process_txt(txt: dict[str, str]) -> tuple[str, str]:
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
    for i in range(len(txt)):
        if txt[i]['from'] == "human":
            qa_pairs += f"[Q] {txt[i]['value']} [/Q] "
        else:
            qa_pairs += f"{txt[i]['value']} "
    return qa_pairs + "<EOS>"


def preprocess_data(data: tuple[Image, str]) -> tuple[Image, str]:
    """Preprocesses the data by extracting questions, answers, and images."""
    img = _process_img(data["image"])  
    txt = _process_txt(data["conversations"]) 
    return img, txt


def compute_clip_score(image: Image, text: str) -> float:
    """Calculates CLIP Score for each image and text"""
    inputs = processor(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        similarity = F.cosine_similarity(image_embeds, text_embeds)
    return similarity.item()