import paddle
from paddle import Tensor

def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = paddle.unbind(x, axis=-1)
    b = paddle.stack([(x_c - 0.5 * w), (y_c - 0.5 * h),
                     (x_c + 0.5 * w), (y_c + 0.5 * h)], axis=-1)
    return b


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = paddle.unbind(x, axis=-1)
    b = paddle.stack([(x0 + x1) / 2, (y0 + y1) / 2,
                     (x1 - x0), (y1 - y0)], axis=-1)
    return b


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = paddle.maximum(boxes1.unsqueeze(1)[:, :, :2], boxes2[:, :2])  # [N,M,2]
    rb = paddle.minimum(boxes1.unsqueeze(1)[:, :, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1.unsqueeze(1) + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = paddle.minimum(boxes1.unsqueeze(1)[:, :, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1.unsqueeze(1)[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks: Tensor) -> Tensor:
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return paddle.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = paddle.arange(0, h, dtype='float32', device=masks.device)
    x = paddle.arange(0, w, dtype='float32', device=masks.device)
    y, x = paddle.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.astype('bool')), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.astype('bool')), 1e8).flatten(1).min(-1)[0]

    return paddle.stack([x_min, y_min, x_max, y_max], 1)
