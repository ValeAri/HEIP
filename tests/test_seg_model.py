import torch
from src.unet import get_seg_model


def test_model_fwdbwd() -> None:
    x = torch.rand([1, 3, 64, 64])

    model = get_seg_model()
    out = model(x)

    out["omnipose"].mean().backward()

    assert out["inst"].shape == torch.Size([1, 2, 64, 64])
    assert out["omnipose"].shape == torch.Size([1, 2, 64, 64])
