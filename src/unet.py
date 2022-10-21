import torch
from typing import List, Dict
from collections import OrderedDict

from cellseg_models_pytorch.models import MultiTaskUnet

__all__ = ["get_seg_model", "convert_state_dict"]


def get_seg_model() -> MultiTaskUnet:
    """Build the cellseg_models_pytorch version of the HEIP segmentation model.

    The model is a multi-task unet that outputs nuclei type segmentation masks
    with Pannuke classes, fg/bg prediction of the nuclei and omnipose regression
    of the nuclei instances.


                    |------ TYPE_DECODER -------- TYPE_HEAD
    ENCODER --------|
                    |------ OMNIPOSE_DECODER ---- OMNIPOSE_HEAD
                    |
                    |------ INSTANCE_DECODER ---- INSTANCE_HEAD


    U-Net architectural choices
    ---------------------------
    - encoder: ImageNet pre-trained Efficient-Net v2 L-variant.
    - long skip: normal u-net style with concatenation
    - short skip: residual
    - normalization: batch channel normalization
    - activation: leaky-relu
    - convolution: weight standardized convolution
    - attention: squeeze and excite

    Returns
    -------
        MultiTaskUnet:
            An initialized multi task U-net model with custom architecture.
    """
    pproc = "omnipose"
    model_parts = [
        "encoder",
        "inst_decoder",
        "inst_seg_head",
        "type_decoder",
        "type_seg_head",
        "omnipose_decoder",
        "omnipose_seg_head",
    ]

    decoders = ("inst", "type", pproc)
    heads = {"inst": {"inst": 2}, "type": {"type": 6}, pproc: {pproc: 2}}

    out_channels = {
        "type": (256, 128, 64, 32, 16),
        "inst": (256, 128, 64, 32, 16),
        pproc: (256, 128, 64, 32, 16),
    }

    n_layers = {
        "inst": (1, 1, 1, 1, 1),
        "type": (1, 1, 1, 1, 1),
        pproc: (1, 1, 1, 1, 1),
    }

    n_blocks = {
        "inst": ((2,), (2,), (2,), (2,), (2,)),
        "type": ((2,), (2,), (2,), (2,), (2,)),
        pproc: ((2,), (2,), (2,), (2,), (2,)),
    }

    long_skips = {"type": "unet", "inst": "unet", pproc: "unet"}

    stage1_deocder = {
        "layer_residual": False,
        "merge_policy": "cat",
        "short_skips": ("basic",),
        "block_types": (("basic_old", "basic_old"),),
        "kernel_sizes": ((3, 3),),
        "expand_ratios": ((1.0, 1.0),),
        "groups": ((1, 1),),
        "biases": ((True, True),),
        "normalizations": (("bcn", "bcn"),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": (("wsconv", "wsconv"),),
        "attentions": ((None, "se"),),
        "preactivates": ((False, False),),
        "preattends": ((False, False),),
        "use_styles": ((False, False),),
    }
    stage2_deocder = {
        "layer_residual": False,
        "merge_policy": "cat",
        "short_skips": ("basic",),
        "block_types": (("basic_old", "basic_old"),),
        "kernel_sizes": ((3, 3),),
        "expand_ratios": ((1.0, 1.0),),
        "groups": ((1, 1),),
        "biases": ((True, True),),
        "normalizations": (("bcn", "bcn"),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": (("wsconv", "wsconv"),),
        "attentions": ((None, "se"),),
        "preactivates": ((False, False),),
        "preattends": ((False, False),),
        "use_styles": ((False, False),),
    }
    stage3_deocder = {
        "layer_residual": False,
        "merge_policy": "cat",
        "short_skips": ("basic",),
        "block_types": (("basic_old", "basic_old"),),
        "kernel_sizes": ((3, 3),),
        "expand_ratios": ((1.0, 1.0),),
        "groups": ((1, 1),),
        "biases": ((True, True),),
        "normalizations": (("bcn", "bcn"),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": (("wsconv", "wsconv"),),
        "attentions": ((None, "se"),),
        "preactivates": ((False, False),),
        "preattends": ((False, False),),
        "use_styles": ((False, False),),
    }
    stage4_deocder = {
        "layer_residual": False,
        "merge_policy": "cat",
        "short_skips": ("basic",),
        "block_types": (("basic_old", "basic_old"),),
        "kernel_sizes": ((3, 3),),
        "expand_ratios": ((1.0, 1.0),),
        "groups": ((1, 1),),
        "biases": ((True, True),),
        "normalizations": (("bcn", "bcn"),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": (("wsconv", "wsconv"),),
        "attentions": ((None, "se"),),
        "preactivates": ((False, False),),
        "preattends": ((False, False),),
        "use_styles": ((False, False),),
    }
    stage5_deocder = {
        "layer_residual": False,
        "merge_policy": "cat",
        "short_skips": ("basic",),
        "block_types": (("basic_old", "basic_old"),),
        "kernel_sizes": ((3, 3),),
        "expand_ratios": ((1.0, 1.0),),
        "groups": ((1, 1),),
        "biases": ((True, True),),
        "normalizations": (("bcn", "bcn"),),
        "activations": (("leaky-relu", "leaky-relu"),),
        "convolutions": (("wsconv", "wsconv"),),
        "attentions": ((None, "se"),),
        "preactivates": ((False, False),),
        "preattends": ((False, False),),
        "use_styles": ((False, False),),
    }

    type_dec_params = (
        stage1_deocder,
        stage2_deocder,
        stage3_deocder,
        stage4_deocder,
        stage5_deocder,
    )

    inst_dec_params = (
        stage1_deocder,
        stage2_deocder,
        stage3_deocder,
        stage4_deocder,
        stage5_deocder,
    )
    pproc_dec_params = (
        stage1_deocder,
        stage2_deocder,
        stage3_deocder,
        stage4_deocder,
        stage5_deocder,
    )

    dec_params = {
        "type": type_dec_params,
        "inst": inst_dec_params,
        pproc: pproc_dec_params,
    }

    model = MultiTaskUnet(
        decoders=decoders,
        heads=heads,
        enc_name="tf_efficientnetv2_l",
        depth=5,
        style_channels=None,
        n_layers=n_layers,
        n_blocks=n_blocks,
        out_channels=out_channels,
        long_skips=long_skips,
        dec_params=dec_params,
        inst_key="inst",
        aux_key=pproc,
    )

    return model, model_parts


def get_ckpt_keys(ckpt: OrderedDict, key: str):
    """Get checkpoint that have a matching key str."""
    keys = []
    for k in ckpt.keys():
        if key in k:
            keys.append(k)

    return keys


def convert_state_dict(
    model_parts: List[str],
    ckpt_empty: Dict[str, torch.Tensor],
    ckpt_trained: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert the Dippa repo trained model state_dict to cellseg_models."""
    state_dict = OrderedDict()

    for k in model_parts:
        keys1 = get_ckpt_keys(ckpt_empty, k)
        keys2 = get_ckpt_keys(ckpt_trained, k)

        for k1, k2 in zip(keys1, keys2):
            state_dict[k1] = ckpt_trained[k2]

    return state_dict
