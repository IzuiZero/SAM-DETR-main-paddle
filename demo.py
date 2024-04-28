import os
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import paddle
from paddle.io import DataLoader
from paddle.vision.transforms import Compose
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--multiscale', default=False, action='store_true')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="dimension of the FFN in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="dimension of the transformer")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads in the transformer attention")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")

    parser.add_argument('--smca', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing. We must use cuda.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint, empty for training from scratch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_every_epoch', default=1, type=int, help='eval every ? epoch')
    parser.add_argument('--save_every_epoch', default=1, type=int, help='save model weights every ? epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):

    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = args.device

    # fix the seed for reproducibility
    seed = args.seed + paddle.distributed.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = paddle.DataParallel(model)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of params in model: ', n_parameters)

    def match_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if "backbone.0" not in n and not match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone.0" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = paddle.optimizer.AdamW(parameters=param_dicts, learning_rate=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_drop)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    #
    # if args.distributed:
    #     sampler_train = paddle.io.DistributedBatchSampler(dataset_train, batch_size=args.batch_size, shuffle=True)
    #     sampler_val = paddle.io.DistributedBatchSampler(dataset_val, batch_size=args.batch_size, shuffle=False)
    # else:
    #     sampler_train = paddle.io.BatchSampler(dataset_train, batch_size=args.batch_size, shuffle=True)
    #     sampler_val = paddle.io.BatchSampler(dataset_val, batch_size=args.batch_size, shuffle=False)

    # data_loader_train = DataLoader(dataset_train,
    #                                batch_sampler=sampler_train,
    #                                collate_fn=utils.collate_fn,
    #                                num_workers=args.num_workers)
    #
    # data_loader_val = DataLoader(dataset_val,
    #                              batch_sampler=sampler_val,
    #                              collate_fn=utils.collate_fn,
    #                              num_workers=args.num_workers)

    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = paddle.load(args.frozen_weights)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = paddle.load(args.resume)
        else:
            checkpoint = paddle.load(args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
            lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    transforms = Compose(make_coco_transforms("val"))
    DETECTION_THRESHOLD = 0.5
    inference_dir = "./images/"
    image_dirs = os.listdir(inference_dir)
    image_dirs = [filename for filename in image_dirs if filename.endswith(".jpg") and 'det_res' not in filename]
    model.eval()
    with paddle.no_grad():
        for image_dir in image_dirs:
            img = Image.open(os.path.join(inference_dir, image_dir)).convert("RGB")
            w, h = img.size
            orig_target_sizes = paddle.to_tensor([[h, w]], dtype='float32')
            img, _ = transforms(img)
            img = img.unsqueeze(0)   # adding batch dimension
            img = img.to(device)
            outputs = model(img)
            results = post_processors['bbox'](outputs, orig_target_sizes)[0]
            indexes = results['scores'] >= DETECTION_THRESHOLD
            scores = results['scores'][indexes]
            labels = results['labels'][indexes]
            boxes = results['boxes'][indexes]

            # Visualize the detection results
            import cv2
            img_det_result = cv2.imread(os.path.join(inference_dir, image_dir))
            for i in range(scores.shape[0]):
                x1, y1, x2, y2 = round(float(boxes[i, 0])), round(float(boxes[i, 1])), round(float(boxes[i, 2])), round(float(boxes[i, 3]))
                img_det_result = cv2.rectangle(img_det_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(inference_dir, "det_res_" + image_dir), img_det_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SAM-DETR", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
