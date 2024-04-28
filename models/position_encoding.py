# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
positional encodings for the transformer.
"""
import math
import paddle
import paddle.nn as nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = paddle.logical_not(mask)
        y_embed = paddle.cumsum(not_mask, axis=1, dtype='float32') - 0.5
        x_embed = paddle.cumsum(not_mask, axis=2, dtype='float32') - 0.5
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps + 0.5) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps + 0.5) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype='float32', device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.concat((paddle.sin(pos_x[:, :, :, 0::2]), paddle.cos(pos_x[:, :, :, 1::2])), axis=3).reshape([pos_x.shape[0], pos_x.shape[1], -1])
        pos_y = paddle.concat((paddle.sin(pos_y[:, :, :, 0::2]), paddle.cos(pos_y[:, :, :, 1::2])), axis=3).reshape([pos_y.shape[0], pos_y.shape[1], -1])
        pos = paddle.concat((pos_y, pos_x), axis=2).transpose([0, 2, 3, 1])
        return pos


@paddle.no_grad()
def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = paddle.arange(128, dtype='float32', device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = paddle.concat((paddle.sin(pos_x[:, :, 0::2]), paddle.cos(pos_x[:, :, 1::2])), axis=2).reshape([pos_x.shape[0], pos_x.shape[1], -1])
    pos_y = paddle.concat((paddle.sin(pos_y[:, :, 0::2]), paddle.cos(pos_y[:, :, 1::2])), axis=2).reshape([pos_y.shape[0], pos_y.shape[1], -1])
    pos = paddle.concat((pos_y, pos_x), axis=2)
    return pos


def build_position_encoding(args):
    if args.position_embedding in ('sine'):
        position_embedding = PositionEmbeddingSine(args.hidden_dim // 2, normalize=True)
    else:
        raise ValueError(f"Unknown args.position_embedding: {args.position_embedding}.")
    return position_embedding
