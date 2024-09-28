# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 34  ##################################################

        self.d_model = d_model
        self.n_levels = n_levels  # 4
        self.n_heads = n_heads
        """From the paper Deformable DETR, the deformable attention module only attends to a small set of key sampling 
        points (K=4) around a reference point, regardless of the spatial size of the feature maps."""
        self.n_points = n_points  # 4

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
                                           Encoder: flattened multi-scale feature maps + positional encodings
                                           Decoder: object queries (nn.Embedding(num_queries, hidden_dim)) + positional
                                                    encodings (object queries -> sigmoid -> reference_points -> spatial PE)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
                                           Encoder: flattened multi-scale feature maps
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                                           tensor([[32, 32], [16, 16], [8, 8], [4, 4]]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                                           tensor([0, 1024=32^2, 1280=32^2+16^2, 1344=32^2+16^2+8^2])
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape  # batch size, 1660 (encoder), num_queries (decoder)
        N, Len_in, _ = input_flatten.shape  # batch size, 1660
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)  # torch.Size([bs, 1360, 8, 32])

        """
        From the paper Deformable DETR, the query feature is used to generate ∆p_mlqk and A_mlqk via linear projection 
        (see Eqn. (3) in the paper Deformable DETR).
        "attention_weights" corresponds to 'A_mlqk' in Eqn. (3) in the paper Deformable DETR.
        "sampling_locations" corresponds to "ɸ_l(p^_q)+Δp_mlqk" in Eqn. (3) in the paper Deformable DETR.
        """
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # Encoder: torch.Size([bs, 1360, num_heads=8, num_feature_levels, num_sampling_points=4, 2])
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # Encoder: torch.Size([bs, 1360, 8, 4x4])
        # Decoder: torch.Size([bs, 2800, 8, 4x4])
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # Encoder: torch.Size([bs, 1360, 8, 4, 4])
        # Decoder: torch.Size([bs, 2800, 8, 4, 4])
        if reference_points.shape[-1] == 2:  # [bs, 1360, num_feature_levels, 2]
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # tensor([[32, 32], [16, 16], [8, 8], [4, 4]])
            """Each query attends to LxK (num_feature_levels*num_sampling_points) sampling points from L multi-scale 
            feature maps."""
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]  # [bs, 1360, 1, num_feature_levels, 1, 2] + [bs, 1360, 8, num_feature_levels, 4, 2]
            # torch.Size([4, 1360, 8, num_feature_levels=4, num_sampling_points=4, 2])
            # In RoI-Transformer:
            # encoder:
            # reference_points: torch.Size([bs*num_proposals, 7*7, num_feature_level=1, 2])
            # sampling_locations: torch.Size([bs*num_proposals, 7*7, num_heads=8, num_feature_level=1, num_sampling_points=4, 2])
            # decoder:
            # reference_points: torch.Size([bs*num_proposals, 30, num_feature_levels=1, 2])
            # sampling_locations: torch.Size([bs*num_proposals, 30, num_heads=8, num_feature_levels=1, num_sampling_points=4, 2])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output
