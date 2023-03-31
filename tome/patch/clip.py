# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/openai/CLIP
# --------------------------------------------------------
# Contributed by https://github.com/FrancescoSaverioZuppichini


from collections import OrderedDict
from typing import Tuple

import torch
from clip.model import LayerNorm, QuickGELU, VisionTransformer, convert_weights
from torch import nn

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

from .timm import ToMeAttention


def convert_attention_block(
    src: nn.MultiheadAttention, dst: ToMeAttention
) -> Tuple[ToMeAttention, torch.device]:
    src_state_dict = src.state_dict()
    dst_state_dict = dst.state_dict()
    src_to_dst_keys = [
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("out_proj.weight", "proj.weight"),
        ("out_proj.bias", "proj.bias"),
    ]

    # for key, data in src_state_dict.items():
    #     print(key, data.dtype)

    for src_key, dst_key in src_to_dst_keys:
        dst_state_dict[dst_key] = src_state_dict[src_key]
    dst.load_state_dict(dst_state_dict)
    src_device = src_state_dict["in_proj_weight"].device
    return dst.to(src_device), src_device


class ToMeResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = ToMeAttention(dim=d_model, num_heads=n_head, qkv_bias=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        x, metric = self.attn(x.permute(1, 0, 2), attn_size)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x.permute(1, 0, 2)

        return x

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)
        # clip uses sequence_lenght first (so batch is the second one)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        x_att, metric = self.attn(self.ln_1(x), attn_size)
        x = x + x_att

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self.mlp(self.ln_2(x))
        return x.permute(1, 0, 2)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.transformer.resblocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for i, resblock in enumerate(model.transformer.resblocks):
        tome_module = ToMeResidualAttentionBlock(
            resblock.attn.embed_dim, resblock.attn.num_heads
        )
        _, device = convert_attention_block(resblock.attn, tome_module.attn)
        tome_module._tome_info = model._tome_info
        tome_module = tome_module.to(device)
        model.transformer.resblocks[i] = tome_module

    # def _convert_weights_to_fp16(l):
    #     if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
    #         l.weight.data = l.weight.data.half()
    #         if l.bias is not None:
    #             l.bias.data = l.bias.data.half()

    #     if isinstance(l, ToMeAttention):
    #         l = l.half()

    # model.apply(_convert_weights_to_fp16)
