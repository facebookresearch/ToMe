# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple, Union

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention
from diffusers.models.unet_blocks import UNetMidBlock2DCrossAttn, CrossAttnUpBlock2D, CrossAttnDownBlock2D
from diffusers.utils import BaseOutput
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from dataclasses import dataclass
from einops import rearrange

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor

class ToMeUpBlock(CrossAttnUpBlock2D):
    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                #print("Up", temb.shape, hidden_states.shape, encoder_hidden_states.shape)
                hidden_states = resnet(hidden_states, temb)
                hidden_states, encoder_hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states, encoder_hidden_states

class ToMeMidBlock(UNetMidBlock2DCrossAttn):
    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            #print("Mid", temb.shape, hidden_states.shape, encoder_hidden_states.shape)
            hidden_states, encoder_hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states, encoder_hidden_states

class ToMeDownBlock(CrossAttnDownBlock2D):
    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                #print("Down", temb.shape, hidden_states.shape, encoder_hidden_states.shape)
                hidden_states = resnet(hidden_states, temb)
                hidden_states, encoder_hidden_states = attn(hidden_states, context=encoder_hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states, encoder_hidden_states

class ToMeSpatialTransformer(SpatialTransformer):
    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        metric = None
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h=height)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual

class ToMeTransformerBlock(BasicTransformerBlock):
    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        hs, metric = self.attn1(self.norm1(hidden_states), size=attn_size)
        hidden_states = hs + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states

class ToMeCrossAttention(CrossAttention):
    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        km = key
        value = self.to_v(context)
        dim = query.shape[-1]
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                key,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, key, self._tome_info["source"]
                )
                self._tome_info["source"] = merge_source(
                    merge, value, self._tome_info["source"]
                )
            key, self._tome_info["size"] = merge_wavg(merge, key)
            value, self._tome_info["size"] = merge_wavg(merge, value)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)


        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # Apply proportional attention
        #size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        #if size is not None:
        #    hidden_states = hidden_states + size.log()[:, None, None, :, 0]
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])
            # Apply proportional attention
            #size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            #if size is not None:
            #    attn_slice = attn_slice + size.log()[start_idx:end_idx, None, None, :, 0]

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

class ToMeUNet(UNet2DConditionModel):
    """
    Modifications:
     - Initialize r, token size, and token sources.
    """

    def forward(self, *args, **kwargs):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        self._tome_info["r"] = parse_r(len([m for m in self.modules() if isinstance(m, BasicTransformerBlock)]), self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        return super().forward(*args, **kwargs)
        

def apply_patch(
    model: UNet2DConditionModel, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """

    model.__class__ = ToMeUNet
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, SpatialTransformer):
            module.__class__ = ToMeSpatialTransformer
        if isinstance(module, BasicTransformerBlock):
            #module.__class__ = ToMeTransformerBlock
            #module._tome_info = model._tome_info
            module.attn1.__class__ = ToMeCrossAttention
            module.attn1._tome_info = model._tome_info
