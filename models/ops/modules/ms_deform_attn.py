import warnings
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Layer):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        self.sampling_offsets.weight.set_value(paddle.zeros_like(self.sampling_offsets.weight))
        thetas = paddle.arange(self.n_heads, dtype='float32') * (2.0 * math.pi / self.n_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / paddle.abs(grid_init).max(-1, keepdim=True)[0]).reshape([self.n_heads, 1, 1, 2]).repeat([1, self.n_levels, self.n_points, 1])
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with paddle.no_grad():
            self.sampling_offsets.bias.set_value(grid_init.reshape([-1]))
        self.attention_weights.weight.set_value(paddle.zeros_like(self.attention_weights.weight))
        self.attention_weights.bias.set_value(paddle.zeros_like(self.attention_weights.bias))
        self.value_proj.weight.set_value(Uniform(-math.sqrt(1 / self.d_model), math.sqrt(1 / self.d_model))(self.value_proj.weight.shape))
        self.value_proj.bias.set_value(paddle.zeros_like(self.value_proj.bias))
        self.output_proj.weight.set_value(Uniform(-math.sqrt(1 / self.d_model), math.sqrt(1 / self.d_model))(self.output_proj.weight.shape))
        self.output_proj.bias.set_value(paddle.zeros_like(self.output_proj.bias))

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = paddle.where(input_padding_mask.unsqueeze(-1), paddle.zeros_like(value), value)
        value = value.reshape([N, Len_in, self.n_heads, self.d_model // self.n_heads])
        sampling_offsets = self.sampling_offsets(query).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points, 2])
        attention_weights = self.attention_weights(query).reshape([N, Len_q, self.n_heads, self.n_levels * self.n_points])
        attention_weights = F.softmax(attention_weights, -1).reshape([N, Len_q, self.n_heads, self.n_levels, self.n_points])
        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack([input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
