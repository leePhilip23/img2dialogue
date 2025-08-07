import pytest
import torch
import torch.nn.functional as F
import requests
from PIL import Image
from utils import model


@pytest.mark.parametrize(
    "img_url, txt_a, txt_b",
    [
        (
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "How many cats are there?",
            "what is the num of cats available?"
        ),
        (
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "Is there a TV in this picture?",
            "there is a tv right?"
        ),
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "What animal is on the image?",
            "what animal is that?"
        ),
    ],
)
def test_invariance(img_url: str, txt_a: str, txt_b: str):
    img = Image.open(requests.get(img_url, stream=True).raw)
    out_a = model.evaluate(img, txt_a).hidden_states
    out_b = model.evaluate(img, txt_b).hidden_states
    cos_sim = F.cosine_similarity(out_a.mean(dim=1), out_b.mean(dim=1), dim=-1)

    # The hidden states should be similar in syntax and semantics
    assert (cos_sim.item() > 0.95).all(), "Invariance: embeddings are not semantically aligned enough"


@pytest.mark.parametrize(
    "img_url, txt_better, txt_worse",
    [
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "What kind of animal is this bear?",
            "what is that"
        ),
        (
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "Is the TV turned on in the image?",
            "is it on"
        ),
    ],
)
def test_directional(img_url: str, txt_better: str, txt_worse: str):
    img = Image.open(requests.get(img_url, stream=True).raw)
    
    better_ids = torch.argmax(model.evaluate(img, txt_better).logits, dim=-1)
    worse_ids = torch.argmax(model.evaluate(img, txt_worse).logits, dim=-1)

    #TODO: Decide between batch decode or just decode for tokenizer
    better_txt = model.tokenizer.decode(better_ids[0], skip_special_tokens=True)
    worse_txt = model.tokenizer.decode(worse_ids[0], skip_special_tokens=True)

    # The more specific or clearer prompt should have more complete answer
    assert len(better_txt) >= len(worse_txt), (
        f"Directional test failed:\n"
        f"Better: {txt_better} -> {better_txt}\n"
        f"Worse: {txt_worse} -> {worse_txt}"
    )


@pytest.mark.parametrize(
    "img_url, prompt, expected_output",
    [
        (
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "How many cats are there?",
            "There are two cats."
        ),
        (
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "Is there a TV in this picture?",
            "Yes"
        ),
        (
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            "What animal is on the image?",
            "It is a bear."
        ),
    ],
)
def test_mft(img_url: str, prompt: str, ground_truth: str):
    img = Image.open(requests.get(img_url, stream=True).raw)
    pred = model.evaluate(img, prompt).hidden_states
    cos_sim = F.cosine_similarity(pred.mean(dim=1), ground_truth.mean(dim=1), dim=-1)
    
    # The output of model should be similar to ground truth
    assert (cos_sim.item() > 0.95).all(), "Minimum Functional Test: embeddings are not semantically aligned enough"