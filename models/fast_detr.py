import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .misc import _get_clones, MLP


class FastDETR(nn.Layer):
    """ This is the SAM-DETR module that performs object detection """
    def __init__(self, args, backbone, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: paddle module of the backbone to be used. See backbone.py
            transformer: paddle module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         that our model can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.num_feature_levels = 3 if self.multiscale else 1          # Hard-coded multiscale parameters
        self.num_queries = args.num_queries
        self.aux_loss = args.aux_loss
        self.hidden_dim = args.hidden_dim
        assert self.hidden_dim == transformer.d_model

        self.backbone = backbone
        self.transformer = transformer

        # Instead of modeling query_embed as learnable parameters in the shape of (num_queries, d_model),
        # we directly model reference boxes in the shape of (num_queries, 4), in the format of (xc yc w h).
        self.query_embed = nn.Embedding(self.num_queries, 4)           # Reference boxes

        # ====================================================================================
        #                                   * Clarification *
        #  -----------------------------------------------------------------------------------
        #  Whether self.input_proj contains nn.GroupNorm should not affect performance much.
        #  nn.GroupNorm() is introduced in some of our experiments by accident.
        #  Our experience shows that it even slightly degrades the performance.
        #  We recommend you to simply delete nn.GroupNorm() in self.input_proj for your own
        #  experiments, but if you wish to reproduce our results, please follow our setups below.
        # ====================================================================================
        if self.multiscale:
            input_proj_list = []
            for _ in range(self.num_feature_levels):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            self.input_proj = nn.LayerList(input_proj_list)
        else:
            if self.args.epochs >= 25:
                self.input_proj = nn.LayerList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    )])
            else:
                self.input_proj = nn.LayerList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )])

        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.set_value(paddle.ones([num_classes]) * bias_value)

        # init bbox_embed
        self.bbox_embed.layers[-1].weight.set_value(paddle.zeros([4, self.hidden_dim]))
        self.bbox_embed.layers[-1].bias.set_value(paddle.zeros([4]))

        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.bbox_embed = _get_clones(self.bbox_embed, args.dec_layers)

        self.transformer.decoder.bbox_embed = self.bbox_embed

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, paddle.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos_embeds = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        hs, reference = self.transformer(srcs, masks, self.query_embed.weight, pos_embeds)

        outputs_coords = []
        outputs_class = []
        for lvl in range(hs.shape[0]):
            reference_before_sigmoid = inverse_sigmoid(reference[lvl])
            bbox_offset = self.bbox_embed[lvl](hs[lvl])
            outputs_coord = (reference_before_sigmoid + bbox_offset).sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_class.append(self.class_embed[lvl](hs[lvl]))
        outputs_coords = paddle.stack(outputs_coords)
        outputs_class = paddle.stack(outputs_class)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords)

        return out

    @paddle.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coords[:-1])]


class SetCriterion(nn.Layer):
    """ This class computes the loss for SAM-DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(src_logits.shape[:2], self.num_classes, dtype=paddle.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = paddle.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) \
                  * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @paddle.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = paddle.as_tensor([len(v["labels"]) for v in targets], dtype=paddle.float32, device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = paddle.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - paddle.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.reshape(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = paddle.concat([paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = paddle.to_tensor([num_boxes], dtype=paddle.float32, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            paddle.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Layer):
    """ This module converts the model's output into the format expected by the coco api"""
    @paddle.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = paddle.topk(prob.view(out_logits.shape[0], -1), 100, axis=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = paddle.gather(boxes, topk_boxes.unsqueeze(-1), axis=1).squeeze(1)

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = paddle.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i in range(out_logits.shape[0]):
            result = {'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]}
            results.append(result)
        return results
