"""
This script defines the head components for the RoI-based polygon prediction model.

1. The `TransformerHead` implements the primary model architecture, based on a transformer.
   Each region-of-interest (RoI) feature is passed through the transformer to predict polygon vertices and classifications.
2. The `TransformerHead` also incorporates a residual learnable logit refinement mechanism.
   This allows the model to iteratively refine the logits layer-by-layer within the transformer, improving the precision of polygon vertex classifications.

For more details, refer to Section 3.2 of the paper.
"""
import copy
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .deformable_transformer import build_deforamble_transformer


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):
    """
    DynamicHead for processing RoI features and making predictions.
    Includes initialization for RoI pooling, backbone feature extraction,
    and building the transformer head with transformer layers.
    """

    def __init__(self, cfg, roi_input_shape):
        """
        Initialize DynamicHead.

        Args:
            cfg: Configuration object.
            roi_input_shape: Dictionary mapping feature levels (p2, p3, etc.) to their ShapeSpec.
        """
        super().__init__()

        # Initialize the RoI Pooler
        self.box_pooler = self._init_box_pooler(cfg, roi_input_shape)

        # Model settings
        num_classes = cfg.MODEL.RoIPoly.NUM_CLASSES
        num_corners = cfg.MODEL.RoIPoly.NUM_CORNERS
        d_model = cfg.MODEL.RoIPoly.HIDDEN_DIM
        num_queries = cfg.MODEL.RoIPoly.NUM_CORNERS

        # Build the transformer head
        transformer = build_deforamble_transformer()
        self.transformer_head = TransformerHead(d_model, num_classes, num_corners, transformer, num_queries)
        self.num_corners = num_corners

        # Focal loss settings
        self.use_focal = cfg.MODEL.RoIPoly.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.RoIPoly.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        # Initialize model parameters
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        # Initialize the RoI box pooler based on configuration.
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # ["p2", "p3", "p4", "p5"]
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION  # 7
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO  # 2
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE  # ROIAlignV2

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type
        )
        return box_pooler

    def forward(self, features, init_bboxes):
        """
        Forward pass through the DynamicHead.

        Args:
            features (list[Tensor]): List of feature maps from backbone.
                                     Each feature map has shape (batch_size, channels, height, width).
            init_bboxes (Tensor): Initial bounding boxes in (x1, y1, x2, y2) format,
                                  shape [batch_size, num_proposals, 4], with absolute coordinates.

        Returns:
        dict: Output containing 'pred_logits' (classification scores) and 'pred_coords' (polygon coordinates).

        Outputs:
        - 'pred_logits' (Tensor): Shape [batch_size, num_proposals, num_queries], class logits.
        - 'pred_coords' (Tensor): Shape [batch_size, num_proposals, num_queries, 2], polygon coordinates.
        """
        # Run the transformer head for feature extraction and prediction
        output = self.transformer_head(features, init_bboxes, self.box_pooler)

        return output


class TransformerHead(nn.Module):
    """
    The TransformerHead class for predicting polygons and their vertex classifications using a transformer-based
    architecture. This head refines the predictions across multiple layers, allowing for fine-tuning of polygon vertices
    and the classification logits for each point (logit embedding).
    """

    def __init__(self, d_model, num_classes, num_corners, transformer, num_queries,
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), aux_loss=False,
                 with_poly_refine=True):
        super().__init__()

        self.d_model = d_model
        self.num_corners = num_corners

        """Initialize transformer head components"""
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # Linear layer for class prediction
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # MLP for predicting polygon coordinates
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        # Query and target embeddings
        self.query_embed = nn.Embedding(num_queries, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        # Logit embedding
        self.logit_embed = nn.Embedding(num_queries, 1)

        self.aux_loss = aux_loss
        self.with_poly_refine = with_poly_refine

        # Initialize class_embed bias and coords_embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)

        """Configure multiple prediction heads based on the transformer layers"""
        num_pred = transformer.decoder.num_layers
        if with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.coords_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.coords_embed = nn.ModuleList([self.coords_embed for _ in range(num_pred)])

        """Attach embedding layers to the transformer decoder"""
        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed

        # The attention mask is used to prevent object queries in one polygon attending to another polygon, default false
        self.attention_mask = None

        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pooler):
        """
        Args:
            features (list[Tensor]): Feature maps from the backbone, list of length 4.
                - p2: Tensor of shape (bs, d_model, w/4, h/4)
                - p3: Tensor of shape (bs, d_model, w/8, h/8)
                - p4: Tensor of shape (bs, d_model, w/16, h/16)
                - p5: Tensor of shape (bs, d_model, w/32, h/32)
            bboxes (Tensor): Proposal bounding boxes of shape (bs, num_proposals, 4),
                             where 4 represents (x1, y1, x2, y2), in absolute coordinates.
            pooler (ROIPooler): ROIAlign layer to pool feature maps for each proposal.

        Returns:
            dict: Dictionary containing predicted vertex classification logits (`pred_logits`) and polygon coordinates
                  (`pred_coords`).
                - 'pred_logits': (bs, num_proposals, num_queries)
                - 'pred_coords': (bs, num_proposals, num_queries, 2)
        """

        # Number of images and proposals
        N, nr_boxes = bboxes.shape[:2]  # N: batch size, nr_boxes: number of proposals

        # ROI feature extraction
        proposal_boxes = [Boxes(bboxes[b]) for b in range(N)]  # Convert bboxes to Boxes format
        roi_features = pooler(features, proposal_boxes)  # Shape: (bs * num_proposals, d_model, 7, 7)

        # Initialize the input for the transformer
        srcs = [roi_features]  # List to hold feature maps for transformer
        masks = [torch.zeros([roi_features.shape[0], roi_features.shape[2], roi_features.shape[3]],
                             device=roi_features.device, dtype=torch.bool)]  # Attention masks, all False
        pos = [build_positional_embedding(roi_features, masks[0], num_pos_feats=128)]  # Positional embeddings

        # Transformer embeddings
        tgt_embeds = self.tgt_embed.weight  # Target embeddings (num_queries, hidden_dim)
        query_embeds = self.query_embed.weight  # Query embeddings (num_queries, 2)
        logit_embeds = self.logit_embed.weight  # Logit embeddings (num_queries, 1)

        # Transformer forward pass
        hs, init_reference, inter_references, inter_classes = self.transformer(bboxes, srcs, masks, pos, query_embeds,
                                                                               tgt_embeds,
                                                                               logit_embeds,
                                                                               self.attention_mask)
        # hs: torch.Size([6, bs * num_proposals, num_queries, d=256])
        # init_reference: torch.Size([bs * num_proposals, num_queries, 2])
        # inter_references: torch.Size([6, bs * num_proposals, num_queries, 2]), range [0, 1]
        # inter_classes: torch.Size([6, bs * num_proposals, num_queries, 1])

        num_layer = hs.shape[0]
        outputs_class = inter_classes.reshape(num_layer, N, nr_boxes, self.num_queries)
        outputs_coord = inter_references.reshape(num_layer, N, nr_boxes, self.num_queries, 2)
        # outputs_class.shape: [6, bs, num_proposals, num_queries]
        # outputs_coord.shape: [6, bs, num_proposals, num_queries, 2]

        # Final output dictionary
        out = {'pred_logits': outputs_class[-1], 'pred_coords': outputs_coord[-1]}

        # Add auxiliary outputs if specified
        if self.aux_loss:  # default: False
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_positional_embedding(x, mask, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
    """
    Builds the positional embeddings for the input feature map, based on sine and cosine functions.

    Args:
        x (torch.Tensor): Input feature map of shape [bs * num_proposals, d_model=256, 7, 7],
        mask (torch.Tensor): Binary mask of shape [bs * num_proposals, 7, 7] indicating valid regions.
        num_pos_feats (int, optional): Number of positional embedding features per spatial location. Default is 128.
        temperature (int, optional): Temperature for scaling the positional encoding frequencies. Default is 10000.
        normalize (bool, optional): Whether to normalize the positional embeddings to the range [0, 1]. Default is True.
        scale (float, optional): Additional scaling factor for the positional embeddings. If None, defaults to 2π.

    Returns:
        torch.Tensor: The positional embeddings for the input feature map, of shape [bs * num_proposals, d_model, 7, 7].
    """
    if scale is None:
        scale = 2 * math.pi
    assert mask is not None
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * scale  # [batch_size, H, W]
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale  # [batch_size, H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0, 1, 2, ..., 127]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t  # [batch_size, H, W, 128]
    pos_y = y_embed[:, :, :, None] / dim_t  # [batch_size, H, W, 128]
    # torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4): torch.Size([bs, H, W, 64, 2])
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    # Shape: torch.Size([bs, H, W, 128])
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    # Shape: torch.Size([bs, H, W, 128])
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # torch.Size([bs, 256, H, W])
    return pos
