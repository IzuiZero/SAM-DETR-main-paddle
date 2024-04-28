import paddle
import paddle.vision.transforms as transforms
from paddle.vision.datasets import CocoDetection
from pycocotools import mask as coco_mask
from pathlib import Path

import datasets.transforms as T


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = paddle.to_tensor(mask, dtype='uint8')
        mask = paddle.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = paddle.stack(masks, axis=0)
    else:
        masks = paddle.zeros((0, height, width), dtype='uint8')
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = paddle.to_tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = paddle.to_tensor(boxes, dtype='float32').reshape([-1, 4])
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clip_(min=0, max=w)
        boxes[:, 1::2].clip_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = paddle.to_tensor(classes, dtype='int64')

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = paddle.to_tensor(keypoints, dtype='float32')
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = paddle.to_tensor([obj["area"] for obj in anno])
        iscrowd = paddle.to_tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = paddle.to_tensor([int(h), int(w)])
        target["size"] = paddle.to_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomSelect(
                transforms.RandomResize(scales, max_size=1333),
                transforms.Compose([
                    transforms.RandomResize([400, 500, 600]),
                    transforms.RandomSizeCrop(384, 600),
                    transforms.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return transforms.Compose([
            transforms.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
