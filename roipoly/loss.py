import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
#from .losses import MaskRasterizationLoss
from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    """ This class computes the loss for multiple polygons.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth polygons and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and coords)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of classes for corner validity (binary)
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # 1, valid or invalid corner
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        #self.raster_loss = MaskRasterizationLoss(None)

    def loss_labels(self, outputs, targets, indices):
        """
        Compute the focal loss for vertex classification.

        Args:
            outputs (dict): The model's output containing 'pred_logits' with shape [bs, num_proposals, num_queries].
            targets (list[dict]): The list of ground truth targets.
            indices (list[tuple]): The list of matching indices between predictions and targets.
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Initialize the target classes tensor with background class (default class 0)
        target_classes = torch.full(src_logits.shape, self.num_classes - 1, dtype=torch.float32,
                                    device=src_logits.device)

        # Assign the ground truth classes (annotated points as class 1, sampled points as class 0)
        target_classes[idx] = target_classes_o

        src_logits = torch.unsqueeze(src_logits.flatten(), -1)
        target_classes = torch.unsqueeze(target_classes.flatten(), -1)

        # sigmoid_focal_loss_jit: (0 for the negative class and 1 for the positive class)
        loss_ce = sigmoid_focal_loss_jit(
                src_logits,  # [bs*num_proposals*num_queries, 1]
                target_classes,  # [bs*num_proposals*num_queries, 1]
                alpha=0.25,
                gamma=2.0,
                reduction="mean",
            )

        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty corners
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([sum(v["lengths"]) for v in targets], device=device) / 2
        # Count the number of predictions that are NOT "no-object" (invalid corners)
        card_pred = (pred_logits.sigmoid() > 0.5).flatten(1, 2).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_polys(self, outputs, targets, indices):
        """Compute the losses related to the polygons:
           1. L1 loss for polygon coordinates
           2. (Optional) Dice loss for polygon rasterizated binary masks

        Args:
            outputs (dict): The model's output containing 'pred_coords' with shape [bs, num_proposals, num_corners, 2].
            targets (list[dict]): The list of ground truth target polygons with 'coords' key.
            indices (list[tuple]): The list of matching indices between predictions and targets.
        """
        assert 'pred_coords' in outputs
        idx = self._get_src_permutation_idx(indices)
        # idx contains the batch ids of matched predicted polygons, the ids of matched predicted polygons in each batch

        src_polys = outputs['pred_coords'][idx]  # torch.Size([num_gt_boxes_batch, num_corners, 2])

        target_polys = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # torch.Size([num_gt_boxes_batch, num_corners*2])

        # Compute L1 loss
        loss_coords = F.l1_loss(
            src_polys.flatten(1, 2),
            target_polys,
            reduction="sum"
        ) / (target_polys.shape[0] * target_polys.shape[1])

        losses = {'loss_coords': loss_coords}

        # Optional: Compute raster-based loss (uncomment the lines below if needed)
        # loss_raster_mask = self.raster_loss(src_polys.flatten(1,2), target_polys, target_len)
        # src_polys.flatten(1,2): torch.Size([num_gt_boxes_batch=81, num_queries_per_poly*2=80])
        # target_polys: # torch.Size([num_gt_boxes_batch=81, num_queries_per_poly*2=80])
        # losses['loss_raster'] = loss_raster_mask

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        # e.g.
        # indices: [(tensor([ 0,  1,  2,  3,  4,  5,  7,  8, 10, 11, 13, 15, 16, 17, 19, 20, 22, 23, 25, 27, 30, 32]), tensor([19,  9,  1,  4,  8,  6, 20, 21,  3,  7, 17, 12,  0, 11,  5, 18, 16, 14, 2, 15, 10, 13])),
        #           (tensor([ 2, 13, 17, 34]), tensor([0, 1, 2, 3])),
        #           (tensor([ 1,  4,  5,  8,  9, 10, 11, 12, 14, 19, 21, 23, 25, 26, 27, 31, 32, 34]), tensor([ 2, 15,  6,  4, 12, 10,  5, 17,  0,  8, 11, 14,  1, 13,  3,  9, 16,  7])),
        #           (tensor([28]), tensor([0]))]
        # batch_idx: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        # 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])  # the batch ids of matched predicted polygons
        # src_idx: tensor([ 0,  1,  2,  3,  4,  5,  7,  8, 10, 11, 13, 15, 16, 17, 19, 20, 22, 23,
        #         25, 27, 30, 32,  2, 13, 17, 34,  1,  4,  5,  8,  9, 10, 11, 12, 14, 19,
        #         21, 23, 25, 26, 27, 31, 32, 34, 28])  # the ids of matched predicted polygons in each batch
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            #'cardinality': self.loss_cardinality,
            'polys': self.loss_polys
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, box_lengths):
        """ This performs the loss computation.
        Args:
            outputs (dict): The model's output, containing the following keys:
                - 'pred_logits': torch.Size([bs, num_proposals, num_corners]), predicted vertex classification logits.
                - 'pred_coords': torch.Size([bs, num_proposals, num_corners, 2]), predicted polygon coordinates.
            targets (list[dict]): Ground truth polygons and classification labels. Each element in the list corresponds to an image in the batch.
            box_lengths (torch.Tensor): Shape [bs], specifying the number of ground truth instances (polygons) in each image in the batch.

        Returns:
            dict: A dictionary containing the computed losses.
        """
        # Filter out auxiliary and encoder outputs
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the predicted and target polygons.
        # Using ground-truth length for matching in this implementation.
        # indices = self.matcher(outputs_without_aux, targets)、
        indices = []
        for box_length in box_lengths:  # Number of ground truth instances per image
            indices.append((
                torch.arange(0, box_length, dtype=torch.int64, device=box_lengths.device),
                torch.arange(0, box_length, dtype=torch.int64, device=box_lengths.device)
            ))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        return losses


# TODO: This class is not actively used in the current pipeline. It can be removed in future code refactoring.
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    We do the matching in polygon (room) level
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_coords: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coords: This is the relative weight of the L1 error of the polygon coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coords = cost_coords
        assert cost_class != 0 or cost_coords != 0, "all costs cant be 0"

    def calculate_angles(self, polygon):
        vect1 = polygon.roll(1, 0)-polygon
        vect2 = polygon.roll(-1, 0)-polygon
        cos_sim = ((vect1 * vect2).sum(1)+1e-9)/(torch.norm(vect1, p=2, dim=1)*torch.norm(vect2, p=2, dim=1)+1e-9)
        # cos_sim = F.cosine_similarity(vect1, vect2)
        # angles = torch.acos(torch.clamp(cos_sim, -1 + 1e-7 , 1 - 1e-7))
        # if torch.isnan(angles).sum() >=1:
        #     print('a')
        # return angles
        return cos_sim

    def calculate_src_angles(self, polygon):
        vect1 = polygon.roll(1, 1)-polygon
        vect2 = polygon.roll(-1, 1)-polygon

        cos_sim = ((vect1 * vect2).sum(-1)+1e-9)/(torch.norm(vect1, p=2, dim=-1)*torch.norm(vect2, p=2, dim=-1)+1e-9)

        return cos_sim

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_polys, num_queries_per_poly] with the classification logits
                 "pred_coords": Tensor of dim [batch_size, num_polys, num_queries_per_poly, 2] with the predicted polygons coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_polys, num_queries_per_poly] (where num_target_polys is the number of ground-truth
                           polygons in the target) containing the class labels
                 "coords": Tensor of dim [num_target_polys, num_queries_per_poly * 2] containing the target polygons coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order), max(index_i) = num_polys - 1
                - index_j is the indices of the corresponding selected targets (in order), max(index_j) = num_target_polys - 1
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_polys, num_target_polys)
        """
        with torch.no_grad():
            bs, num_polys = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            src_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            # torch.Size([batch_size*num_polys, num_queries_per_poly]) = torch.Size([140, 80])
            src_polys = outputs["pred_coords"].flatten(0, 1).flatten(1, 2)
            # torch.Size([batch_size*num_polys, num_queries_per_poly*2]) = torch.Size([140, 160])

            # Also concat the target labels and coords
            tgt_ids = torch.cat([v["labels"] for v in targets])
            # torch.Size([num_gt_polys_per_batch, num_queries_per_poly]) = torch.Size([43, 80])
            # the annotated points are marked as class 1, the padded points are marked as class 0
            tgt_polys = torch.cat([v["coords"] for v in targets])
            # torch.Size([num_gt_polys_per_batch, num_queries_per_poly*2]) = torch.Size([43, 160])
            tgt_len = torch.cat([v["lengths"] for v in targets])
            # torch.Size([num_gt_polys_per_batch]) = torch.Size([43])

            # Compute the pair-wise classification cost.
            # We just use the L1 distance between prediction probality and target labels (inc. no-object calss)
            cost_class = torch.cdist(src_prob, tgt_ids, p=1)

            # Compute the L1 cost between coords
            # Here we do not consider no-object corner in target since we filter out no-object corners in results
            cost_coords = torch.zeros([src_polys.shape[0], tgt_polys.shape[0]], device=src_polys.device)
            # torch.Size([batch_size*num_polys, num_gt_polys_per_batch]) = torch.Size([140, 43])
            for i in range(tgt_polys.shape[0]):
                tgt_polys_single = tgt_polys[i, :tgt_len[i]]  # torch.Size([num_gt_corners_per_poly*2]) = e.g. torch.Size([10])
                # Get all possible permutation of a polygon
                all_polys = get_all_order_corners(tgt_polys_single)  # torch.Size([num_gt_corners_per_poly, num_gt_corners_per_poly*2]) = e.g. torch.Size([5, 10])
                cost_coords[:, i] = torch.cdist(src_polys[:, :tgt_len[i]], all_polys, p=1).min(axis=-1)[0]  # [140, 1]
                # torch.cdist(src_polys[:, :tgt_len[i]], all_polys, p=1): [batch_size*num_polys, num_gt_corners_per_poly] = [140, 5]

            # Final cost matrix
            C = self.cost_coords * cost_coords + self.cost_class * cost_class
            C = C.view(bs, num_polys, -1).cpu()

            sizes = [len(v["coords"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]