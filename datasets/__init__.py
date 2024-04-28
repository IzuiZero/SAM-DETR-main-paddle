import paddle.vision.datasets as datasets
import pycocotools.coco as coco

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, datasets.CocoDetection):
            break
        if isinstance(dataset, paddle.io.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
