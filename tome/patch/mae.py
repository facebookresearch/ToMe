# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# mae: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.utils import parse_r

from .timm import ToMeBlock, ToMeAttention


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - For MAE: make global average pooling proportional to token size
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # From the MAE implementation
            B = x.shape[0]
            x = self.patch_embed(x)

            T = x.shape[1]

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            if self.global_pool:
                # ---- ToMe changes this ----
                # Global average pool proportional to token size
                if self._tome_info["size"] is not None:
                    x = (x * self._tome_info["size"])[:, 1:, :].sum(dim=1) / T
                else:
                    x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                # ---- End of change ----

                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False
):
    """
    Applies ToMe to this MAE transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For MAE models, prop_attn should be set to false.
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
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
