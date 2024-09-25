# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------
"""
DeformableTransformer class for building the deformable transformer architecture.

This class implements both the encoder and decoder of a deformable transformer model for RoI-based polygon prediction tasks,
enabling layer-by-layer refinement of both polygon vertex coordinates and vertex classification logits.

Attributes:
    d_model: The dimension of the model.
    nhead: Number of attention heads.
    num_encoder_layers: Number of layers in the transformer encoder.
    num_decoder_layers: Number of layers in the transformer decoder.
    dim_feedforward: Size of the feedforward layers in the model.
    dropout: Dropout rate.
    activation: Activation function type (e.g., "relu").
    poly_refine: Whether to use polygon refinement in the decoder.
    return_intermediate_dec: Whether to return intermediate outputs from the decoder layers.
    aux_loss: Whether to use auxiliary loss.
    num_feature_levels: Number of feature levels for multi-scale processing.
    dec_n_points: Number of sampling points for decoder.
    enc_n_points: Number of sampling points for encoder.
    query_pos_type: Type of positional embedding for the queries.
"""
import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
import argparse
import numpy as np
import time
import cv2, json
from head import MLP


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", poly_refine=True, return_intermediate_dec=False, aux_loss=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, query_pos_type="none"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Build encoder
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # Build decoder
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, poly_refine,
                                                    return_intermediate_dec, aux_loss, query_pos_type)

        # Multi-scale feature embeddings
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))  # torch.Size([4, 256])

        if query_pos_type == 'sine':
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize model parameters. Applies Xavier uniform initialization to layers
        and normal distribution to level embedding. Resets parameters for MSDeformAttn layers.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """
        Calculate valid ratio of non-masked elements in height and width dimensions of the mask.
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)  # torch.Size([bs]) (~mask[:, :, 0]: torch.Size([bs, H]))
        valid_W = torch.sum(~mask[:, 0, :], 1)  # torch.Size([bs]) (~mask[:, 0, :]: torch.Size([bs, H]))
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # torch.Size([bs, 2])
        return valid_ratio

    def forward(self, bboxes, srcs, masks, pos_embeds, query_embed=None, tgt=None, logit_embed=None, tgt_masks=None):
        """
        Args:
            bboxes (Tensor): Tensor of shape [N=bs, nr_boxes=num_proposals, 4], (x1, y1, x2, y2), absolute coordinates.
            srcs (list[Tensor]): List of feature maps from different levels, shapes:
                - torch.Size([bs, 256, H0/8, W0/8]),
                - torch.Size([bs, 256, H0/16, W0/16]),
                - torch.Size([bs, 256, H0/32, W0/32]),
                - torch.Size([bs, 256, H0/64, W0/64]).
                Here, srcs will be [torch.Size([bs*num_proposals, 256, 7, 7])].
            masks (list[Tensor]): List of masks corresponding to the feature maps, shapes:
                - torch.Size([bs, H0/8, W0/8]),
                - torch.Size([bs, H0/16, W0/16]),
                - torch.Size([bs, H0/32, W0/32]),
                - torch.Size([bs, H0/64, W0/64]).
                Here, masks will be [torch.Size([bs*num_proposals, 7, 7])].
            pos_embeds (list[Tensor]): Positional embeddings for each feature level, shapes:
                - torch.Size([bs, 256, H0/8, W0/8]),
                - torch.Size([bs, 256, H0/16, W0/16]),
                - torch.Size([bs, 256, H0/32, W0/32]),
                - torch.Size([bs, 256, H0/64, H0/64]).
                Here, pos_embeds will be [torch.Size([bs*num_proposals, 256, 7, 7])].
            query_embed (Tensor): Query embedding of shape nn.Embedding(num_queries, 2).weight.
            tgt (Tensor): Target embedding of shape nn.Embedding(num_queries, hidden_dim).weight.
            logit_embed (Tensor): Logit embedding of shape nn.Embedding(num_queries, 1).weight.
            tgt_masks (Tensor): Optional target masks.

        Returns:
        tuple: Contains:
            - hs (Tensor): Output of the decoder, shape [6, bs*num_proposals, num_queries, d=256].
            - init_reference_out (Tensor): Initial reference points, shape [bs*num_proposals, num_queries, 2], range [0, 1].
            - inter_references (Tensor): Intermediate reference points, shape [6, bs*num_proposals, num_queries, 2].
            - inter_classes (Tensor): Intermediate classification logits, shape [6, bs*num_proposals, num_queries, 1].
        """
        assert query_embed is not None

        # Flatten input feature maps and prepare for the encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = src.flatten(2).transpose(1, 2)  # Flatten spatial dimensions, shape [bs*num_proposals, HW=7*7, d]
            mask = mask.flatten(1)  # Flatten mask, shape [bs, HW]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # Flatten positional embeddings, shape [bs*num_proposals, HW, d]

            # To identify which feature level each query pixel lies in, we add scale-level embedding. to the positional embedding
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # [bs*num_proposals, HW, d] + [1, 1, d]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)  # shape [bs*num_proposals, HW, d]
        mask_flatten = torch.cat(mask_flatten, 1)  # shape [bs*num_proposals, HW]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # shape [bs*num_proposals, HW, d]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder pass
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # Prepare input for decoder
        bs, _, c = memory.shape  # bs*num_proposals, 1*7*7, 256

        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # shape [bs*num_proposals, num_queries, 2]
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)  # shape [bs*num_proposals, num_queries, d=256]
        """From the paper Deformable DETR: For each object query, the 2-d normalized coordinate of the reference point
        # p^_q is predicted from its object query embedding via a learnable linear projection followed by a sigmoid function."""
        reference_points = query_embed.sigmoid()  # shape [bs*num_proposals, num_queries, 2]), range from 0 to 1
        init_reference_out = reference_points
        logit_embed = logit_embed.unsqueeze(0).expand(bs, -1, -1)  # shape [bs*num_proposals, num_queries, 1]

        # Decoder pass
        hs, inter_references, inter_classes = self.decoder(bboxes, tgt, reference_points, memory, spatial_shapes,
                                                           level_start_index, valid_ratios, logit_embed,
                                                           query_pos=query_embed, src_padding_mask=mask_flatten,
                                                           tgt_masks=tgt_masks)

        return hs, init_reference_out, inter_references, inter_classes


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        Args:
            src (Tensor): Flattened RoI feature, shape [bs*num_proposals, 1*7*7, d=256].
            pos (Tensor): 2D spatial positional encoding combined with feature map level encoding, shape [bs*num_proposals, 1*7*7, d].
            reference_points (Tensor): Normalized reference points for each feature level, shape [bs*num_proposals, 1*7*7, num_feature_levels, 2].
            spatial_shapes (Tensor): Spatial shapes of the feature maps, tensor [[7, 7]].
            level_start_index (Tensor): Start index of each feature level, tensor [0].
            padding_mask (Tensor, optional): Padding mask indicating True for padding elements, False for non-padding elements, shape [bs*num_proposals, 1*7*7].

        Returns:
            Tensor: Updated feature after self-attention and feed-forward network, shape [bs*num_proposals, 1*7*7, d].
        """
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # torch.Size([batch size, H_^2])
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)  # torch.Size([batch size, W_^2])
            ref = torch.stack((ref_x, ref_y), -1)  # torch.Size([batch size, H_^2, 2])
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # torch.Size([batch size, H0^2+H1^2+H2^2+H3^2, 2])
        """
        First, we sample K sampling points round the reference point for each feature map. For each query pixel, the 
        reference point is itself. Then each query attends to these LxK sampling points.
        "reference_points"[:, i, 0:2] is the normalized 2-D spatial position of the i-th query pixl in its 
        corresponding feature map. Since each query attends to LxK sampling points from L feature maps, these normalized
        2-D spatial positions need to be multiplied by "valid_ratios" of L feature maps.  
        """
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # shape [bs, 1360, 1, 2] x [bs, 1, num_feature_levels, 2]
        """
        "reference_points"[:, i, 0:4, 0:2] are the normalized 2-D spatial positions of the i-th query in L multi-scale 
        feature maps (Each has taken into account the effect of image padding).
        """
        return reference_points  # shape [bs, 1360, num_feature_levels, 2]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)  # [bs, 1360, num_feature_levels, 2]
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # block logit mlp.
        self.block_logit_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))  # 256 -> 1024 -> 256
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def with_logit_embed(self, tensor, logit_emb):  ##################################################logit_embed
        """
        Integrates the learnable logit embedding to the query tensor using the adaLN modulation mechanism,
        as proposed in DiT (Diffusion Transformer).

        Args:
            tensor (Tensor): Query features, shape [bs*num_proposals, num_corners, 256].
            logit_emb (Tensor): Learnable logit embedding, shape [bs*num_proposals, num_queries, 1024].

        Returns:
            Tensor: Query features modulated by the logit embedding, shape [bs*num_proposals, num_corners, 256].
        """
        # Pass logit embedding through an MLP, splitting into scale and shift vectors.
        N, num_queries = tensor.shape[0], tensor.shape[1]
        scale_shift = self.block_logit_mlp(logit_emb)  # [bs*num_proposals, num_corners, 1024] -> [bs*num_proposals, num_corners, 512]
        scale_shift = scale_shift.flatten(0, 1)
        scale, shift = scale_shift.chunk(2, dim=1)

        # Apply adaptive layer normalization (adaLN) style modulation
        tensor = tensor.flatten(0, 1)
        tensor = tensor * (scale + 1) + shift
        tensor = tensor.view(N, num_queries, -1)  # [bs*num_proposals, num_corners, 256]
        return tensor

    def forward(self, tgt, query_pos, logit_embed, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, tgt_masks=None):
        """
        Args:
            tgt (torch.Tensor): Decoder embeddings or object queries.
                                Shape: [bs*num_proposals, num_queries, d=256].
                                Initialized as nn.Embedding(num_queries, hidden_dim).
            query_pos (torch.Tensor): Spatial positional encodings of reference points.
                                      Shape: [bs*num_proposals, num_queries, d=256].
            logit_embed (torch.Tensor): Logit embeddings.
                                        Shape: [bs*num_proposals, num_queries, 1024].
            reference_points (torch.Tensor): Normalized 2D coordinates of reference points.
                                             Shape: [bs*num_proposals, num_queries, num_feature_levels, 2].
            src (torch.Tensor): Flattened multi-scale feature maps from the encoder.
                                Shape: [bs*num_proposals, 1*7*7, d=256].
            src_spatial_shapes (torch.Tensor): Spatial shapes of feature maps.
                                               Shape: tensor([[7, 7]]).
            level_start_index (torch.Tensor): Starting index of each feature map level in the flattened feature map.
                                              Shape: tensor([0]).
            src_padding_mask (torch.Tensor, optional): Padding mask for the feature maps. True for padding elements,
                                                       False for non-padding elements. Default: None.
                                                       Shape: [bs*num_proposals, 1*7*7].
            tgt_masks (torch.Tensor, optional): Mask for the decoder attention layers. Default: None.

        Returns:
            torch.Tensor: Updated decoder output after self-attention, cross-attention, and feed-forward network.
                          Shape: [bs*num_proposals, num_queries, d=256].
        """
        # Self-attention with positional encodings and logit embeddings.
        q = k = self.with_pos_embed(tgt, query_pos)
        q = k = self.with_logit_embed(q, logit_embed)

        # Self-attention layer
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_masks)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Cross-attention with updated tgt and reference points
        tgt2 = self.cross_attn(self.with_logit_embed(self.with_pos_embed(tgt, query_pos), logit_embed),
                               reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, poly_refine=True, return_intermediate=False, aux_loss=False, query_pos_type='none'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.logit_refine = True  # Enables layer-by-layer refinement of classification logits
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type
        
        self.coords_embed = None  # MLP(hidden_dim, hidden_dim, 2, 3)
        self.class_embed = None  # nn.Linear(hidden_dim, num_classes)  # num_classes=1
        self.pos_trans = None  # nn.Linear(d_model, d_model)
        self.pos_trans_norm = None  # nn.LayerNorm(d_model)

        # Define the learnable logit MLP, similar to diffusion models' timestep embedding
        self.d_model = 256
        logit_dim = self.d_model * 4
        self.logit_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.d_model),
            nn.Linear(self.d_model, logit_dim),
            nn.GELU(),  # nn.SiLU() in DiT/models.py/class TimestepEmbedder(nn.Module):
            nn.Linear(logit_dim, logit_dim),  # [bs, 1024]
        )

    def get_query_pos_embed(self, ref_points):
        """
        Generates sinusoidal positional embeddings for the reference points.
        """
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) # [128]
        ref_points = ref_points * scale
        pos = ref_points[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, bboxes, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios, logit_embed, query_pos=None, src_padding_mask=None, tgt_masks=None):
        """
        Args:
            bboxes (torch.Tensor): Bounding boxes of RoIs, shape [bs, num_proposals, 4], xyxy, absolute coordinates.
            tgt (torch.Tensor): Decoder embeddings or object queries, shape [bs*num_proposals, num_queries, d=256].
            logit_embed (torch.Tensor): Initial classification logits, shape [bs*num_proposals, num_queries, 1].
            reference_points (torch.Tensor): Normalized reference points, shape [bs*num_proposals, num_queries, 2].
            src (torch.Tensor): Encoder output, shape [bs*num_proposals, 1*7*7, d=256].
            src_spatial_shapes (torch.Tensor): Spatial shapes of feature maps, shape tensor([[7, 7]]).
            src_level_start_index (torch.Tensor): Start index of each level in the flattened feature maps.
            src_valid_ratios (torch.Tensor): Valid ratios of feature maps, shape [bs*num_proposals, num_feature_levels=1, 2].
            src_padding_mask (torch.Tensor, optional): Padding mask, True for padding elements. Default: None.
            tgt_masks (torch.Tensor, optional): Attention masks for the decoder. Default: None.

        Returns:
            torch.Tensor: Updated decoder outputs after refinement of vertex coordinates and classification logits.
        """
        output = tgt    # [bs*num_proposals, num_queries, d=256]
        _, num_queries, _ = tgt.shape

        # Compute relative reference points within the RoI bounding box (x1y1 is top-left, wh is width-height)
        x1y1 = bboxes.flatten(0, 1)[:, 0:2]  # [bs*num_proposals, 2], absolute coordinates
        x2y2 = bboxes.flatten(0, 1)[:, 2:]  # [bs*num_proposals, 2], absolute coordinates
        wh = x2y2 - x1y1  # [bs*num_proposals, 2], width-height of bounding boxes
        x1y1 = x1y1[:, None, :].expand(-1, num_queries, -1)  # [bs*num_proposals, num_queries, 2]
        wh = wh[:, None, :].expand(-1, num_queries, -1)  # [bs*num_proposals, num_queries, 2]

        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []

        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]  # [bs*num_proposals, num_queries, num_feature_levels, 2]
            
            if self.query_pos_type == 'sine':
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))  # [bs*num_proposals, num_queries, d=256]

            elif self.query_pos_type == 'none':
                query_pos = None

            # Generate logit embedding using the MLP
            logit_embedding = self.logit_mlp(logit_embed.sigmoid())  # [bs*num_proposals, num_queries, 1024]

            output = layer(output, query_pos, logit_embedding, reference_points_input, src,
                           src_spatial_shapes, src_level_start_index, src_padding_mask, tgt_masks)
            # output: object queries, torch.Size([bs*num_proposals, num_queries, d=256])

            # Refine vertex coordinates layer-by-layer
            if self.poly_refine:  # True
                offset = self.coords_embed[lid](output)  # [bs*num_proposals, num_queries, 2], absolute
                assert reference_points.shape[-1] == 2
                new_reference_points = offset + inverse_sigmoid(reference_points)  # [bs*num_proposals, num_queries, 2]
                reference_points = new_reference_points.sigmoid()  # range [0, 1]
            # If not using iterative polygon refinement, just output the reference points decoded from the last layer
            elif lid == len(self.layers)-1:
                offset = self.coords_embed[-1](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()

            # Refine logits layer-by-layer if logit refinement is enabled
            if self.logit_refine:
                logit_offset = self.class_embed[lid](output)  # [bs*num_proposals, num_queries, 1]
                logit_embed = logit_embed + logit_offset  # Residual logit refinement
            # Otherwise, we only predict class label from the last layer
            elif lid == len(self.layers)-1:
                logit_offset = self.class_embed[-1](output)  # [bs*num_proposals, num_queries, 1]
                logit_embed = logit_embed + logit_offset

            # Compute global reference points (relative to the image)
            global_reference_points = reference_points * wh + x1y1  # Absolute coordinates for loss calculation

            if self.return_intermediate:  # True
                intermediate.append(output)
                intermediate_reference_points.append(global_reference_points)
                intermediate_classes.append(logit_embed)

        if self.return_intermediate:  # True
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_classes)

        return output, global_reference_points, logit_embed


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # d_model=256

    def forward(self, logits):
        """
        the same as DiT/models.py/class TimestepEmbedder(nn.Module):/def timestep_embedding(t, dim, max_period=10000):
        """
        device = logits.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)  # torch.Size([128])

        embeddings = logits * embeddings[None, None, :]  # [bs * num_proposals, num_queries, 1] * [1, 1, 128]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # torch.Size([bs * num_proposals, num_queries, 256])


def build_deforamble_transformer():
    return DeformableTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        poly_refine=True,
        return_intermediate_dec=True,
        aux_loss=False,
        num_feature_levels=1,  # Since we are using RoIAlign to extract RoI features from each bounding box,
                               # we only extract the RoI feature from one feature map. Thus, we only have one feature level,
                               # making num_feature_levels equal to 1.
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type='sine')


