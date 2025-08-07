import random
import pytest
from pytest import FixtureRequest
from datasets import load_dataset

@pytest.fixture(scope="module", params=["train", "val"])
def sampled_data(request: FixtureRequest) -> list:
    dataset = load_dataset("lmms-lab/LLaVA-NeXT-Data", split=request.param)
    return random.sample(list(dataset), k=20)