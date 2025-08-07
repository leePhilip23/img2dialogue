import pytest
from utils import preprocess_data, compute_clip_score


@pytest.mark.parametrize("threshold", [0.7])
def test_clip_score_alignment(sample_data: list[dict], threshold: float) -> None:
    failures = 0
    for img_txt in sample_data:
        img, conversation = preprocess_data(img_txt)

        assert img, "Image data is empty"
        assert conversation, "Text data is empty"

        score = compute_clip_score(img, conversation)

        if score < threshold:
            failures += 1

    # Make sure image and conversation are aligned
    assert failures < 1, f"{failures} examples has CLIP score less than {threshold}"
