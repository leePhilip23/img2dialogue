import pytest
import torch
import requests
from PIL import Image
from src.utils.specs import Retrieve


@pytest.mark.parametrize(
    "img_url, txt",
    [
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "What kind of animal is this bear?"
        ),
        (
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "Is the TV turned on in the image?"
        ),
    ],
)
def test_directional(img_url: str, txt: str) -> None:
    img = Image.open(requests.get(img_url, stream=True).raw)
    device = Retrieve.get_device()
    model = Retrieve.get_model(device)
    
    tokens = torch.argmax(model.evaluate(img, txt).logits, dim=-1)
    output = model.tokenizer.decode(tokens[0], skip_special_tokens=True)

    assert isinstance(output, str), "Did not output a response correcty"