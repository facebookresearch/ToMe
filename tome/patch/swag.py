# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SWAG: https://github.com/facebookresearch/SWAG
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

# Since we don't necessarily have the swag code available, this patch is a little bit more involved


def make_block_class(block_cls):
    class ToMeBlock(block_cls):
        """
        Modifications:
        - Apply ToMe between the attention and mlp blocks
        - Compute and propogate token size and potentially the token sources.
        """

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Note: this is copied from swag.models.vision_transformer.EncoderBlock with modifications.
            x = self.ln_1(input)
            attn_size = (
                self._tome_info["size"] if self._tome_info["prop_attn"] else None
            )
            x_attn, metric = self.self_attention(x, size=attn_size)
            x = self.dropout(x_attn)
            x = x + input

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
                x, self._tome_info["size"] = merge_wavg(
                    merge, x, self._tome_info["size"]
                )

            y = self.ln_2(x)
            y = self.mlp(y)
            return x + y

    return ToMeBlock


class ToMeAttention(torch.nn.MultiheadAttention):
    """
    Modifications:
    - Apply proportional attention
    - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        # Return k as well here
        return x, k.mean(1)


def make_transformer_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.encoder.layers), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def make_encoder_class(encoder_class):
    class ToMeEncoder(encoder_class):
        """
        Modifications:
        - Permute encoder dims so it's (batch, tokens, channels).
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pos_embedding

            x = x.transpose(0, 1)
            x = self.ln(self.layers(self.dropout(x)))
            x = x.transpose(0, 1)
            return x

    return ToMeEncoder


def apply_patch(model, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    if model.__class__.__name__ == "ToMeVisionTransformer":
        # This model was already patched!
        return

    EncoderClass = None
    BlockClass = None
    TransformerClass = model.__class__

    # Collect class names
    for module in model.modules():
        if module.__class__.__name__ == "EncoderBlock":
            BlockClass = module.__class__
        elif module.__class__.__name__ == "Encoder":
            EncoderClass = module.__class__

    if BlockClass is None or EncoderClass is None:
        print(
            "Error patching model: this model isn't a SWAG transformer or the interface has been updated."
        )
        return

    ToMeBlock = make_block_class(BlockClass)
    ToMeEncoder = make_encoder_class(EncoderClass)
    ToMeVisionTransformer = make_transformer_class(TransformerClass)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.classifier == "token",
        "distill_token": False,
    }

    for module in model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.__class__ = ToMeAttention
        elif isinstance(module, EncoderClass):
            module.__class__ = ToMeEncoder
