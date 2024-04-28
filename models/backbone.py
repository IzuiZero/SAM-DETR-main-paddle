import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import torchvision
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = paddle.ones([n], dtype='float32')
        self.bias = paddle.zeros([n], dtype='float32')
        self.running_mean = paddle.zeros([n], dtype='float32')
        self.running_var = paddle.ones([n], dtype='float32')

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape([1, -1, 1, 1])
        b = self.bias.reshape([1, -1, 1, 1])
        rv = self.running_var.reshape([1, -1, 1, 1])
        rm = self.running_mean.reshape([1, -1, 1, 1])
        eps = 1e-5
        scale = w * (rv + eps).sqrt().reciprocal()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Layer):
    def __init__(self, backbone: nn.Layer, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.stop_gradient = True
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            # Hard-coded backbone parameters
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            # Hard-coded backbone parameters
            self.strides = [32]
            self.num_channels = [2048]
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m.unsqueeze(0).astype('float32'), size=x.shape[-2:]).to(paddle.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        assert name not in ('resnet18', 'resnet34'), "Number of channels are hard coded, thus do not support res18/34."
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).astype(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or args.multiscale
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
