# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Layer):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.stop_gradient = True

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, paddle.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.reshape([bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1]])

        out["pred_masks"] = outputs_seg_masks
        return out


def _expand(tensor, length: int):
    return paddle.unsqueeze(tensor, axis=1).tile([1, int(length), 1, 1, 1]).flatten(0, 1)


class MaskHeadSmallConv(nn.Layer):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = nn.Conv2D(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2D(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2D(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2D(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.lay5 = nn.Conv2D(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(8, inter_dims[4])
        self.out_lay = nn.Conv2D(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = nn.Conv2D(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2D(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2D(fpn_dims[2], inter_dims[3], 1)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingUniform_(m.weight, a=1)
                nn.initializer.Constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = paddle.concat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Layer):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias_attr=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias_attr=bias)

        nn.initializer.Constant_(self.k_linear.bias, 0)
        nn.initializer.Constant_(self.q_linear.bias, 0)
        nn.initializer.XavierUniform_(self.k_linear.weight)
        nn.initializer.XavierUniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.reshape(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = paddle.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), axis=-1).reshape(weights.shape)
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Layer):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @paddle.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].numpy().tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].astype('float32'), size=tuple(tt.numpy().tolist()), mode="nearest"
            ).astype('uint8')

        return results


class PostProcessPanoptic(nn.Layer):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.numpy().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.numpy() != (outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = paddle.zeros((h, w), dtype=paddle.int64, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id == eq_id, equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)
                m_id = rgb2id(seg_img)

                area = paddle.bincount(m_id.flatten())
                area[0] = 0
                return m_id, area

            m_id_foreground, area_foreground = get_ids_area(cur_masks, cur_scores)

            # Post process things: We want to make sure than thing masks don't overlap
            # This is achieved by computing the intersection between all pairs of masks
            # and if two masks have more than 50% overlap, we keep the one with the highest score
            things_idx = (cur_classes >= len(self.is_thing_map)).nonzero(as_tuple=False).squeeze(1)
            if len(things_idx) > 0:
                cur_boxes_things = cur_boxes[things_idx]
                cur_masks_things = cur_masks[things_idx]
                cur_scores_things = cur_scores[things_idx]
                cur_classes_things = cur_classes[things_idx]

                # Compute pairwise IoU for things
                pairwise_iou = box_ops.box_iou(cur_boxes_things, cur_boxes_things)
                pairwise_iou = paddle.triu(pairwise_iou, diagonal=1)
                pairs_to_delete = (pairwise_iou > 0.5).nonzero(as_tuple=False)
                to_delete = pairs_to_delete[:, 1]

                # mask_to_delete = (cur_masks_things.new_zeros(cur_masks_things.shape[0]).scatter_(0, to_delete, 1) == 1)
                # keep = ~mask_to_delete
                keep = paddle.ones_like(cur_masks_things, dtype=paddle.bool)
                keep[to_delete] = False
                cur_boxes_things = cur_boxes_things[keep]
                cur_masks_things = cur_masks_things[keep]
                cur_scores_things = cur_scores_things[keep]
                cur_classes_things = cur_classes_things[keep]

                m_id_things, area_things = get_ids_area(cur_masks_things, cur_scores_things, dedup=True)
            else:
                cur_boxes_things = cur_masks_things = cur_scores_things = cur_classes_things = m_id_things = area_things = None

            # We merge both segmentations into a single one
            # Note: "stuff" classes are supposed to have been merged in the first part
            if m_id_things is not None:
                m_id_final = paddle.where(m_id_things == 0, m_id_foreground, m_id_things)
            else:
                m_id_final = m_id_foreground

            # This is the final set of labels
            # For stuff classes, labels were already correct
            # For things classes, we relabel them to have contiguous IDs (required by the panoptic API)
            m_id_final_flat = m_id_final.flatten()
            if cur_classes_things is not None:
                for new_id, old_id in enumerate((cur_classes_things - len(self.is_thing_map)).unique()):
                    m_id_final_flat[m_id_final_flat == old_id] = new_id + 1  # +1 because class 0 is "stuff"
            m_id_final = m_id_final_flat.view_as(m_id_final)

            if len(m_id_things) > 0:
                assert m_id_final.max() < 183, f"Problem in {size} : {m_id_final.max()} {m_id_things.max()} {m_id_foreground.max()}"

            preds.append({"segments_info": area_things, "segments": m_id_final, "instances": cur_boxes_things})
        return preds


# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
