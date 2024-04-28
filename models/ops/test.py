import time
import paddle
import paddle.nn as nn
from paddle.autograd import gradcheck
import numpy as np

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype='int64')
level_start_index = paddle.concat((paddle.zeros((1, ), dtype='int64'), paddle.cumsum(paddle.prod(shapes, axis=1))[:-1]))
S = sum([(H*W).item() for H, W in shapes.numpy()])


paddle.manual_seed(3)


@paddle.no_grad()
def check_forward_equal_with_pytorch_double():
    value = paddle.rand([N, S, M, D], dtype='float64') * 0.01
    sampling_locations = paddle.rand([N, Lq, M, L, P, 2], dtype='float64')
    attention_weights = paddle.rand([N, Lq, M, L, P], dtype='float64') + 1e-5
    attention_weights /= attention_weights.sum(axis=(-1), keepdim=True).sum(axis=(-2), keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.numpy().astype(np.float64), shapes.numpy(), sampling_locations.numpy().astype(np.float64), attention_weights.numpy().astype(np.float64)).detach().numpy()
    output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(), im2col_step).detach().numpy()
    fwdok = np.allclose(output_cuda, output_pytorch)
    max_abs_err = np.max(np.abs(output_cuda - output_pytorch))
    max_rel_err = np.max(np.abs((output_cuda - output_pytorch) / output_pytorch))

    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@paddle.no_grad()
def check_forward_equal_with_pytorch_float():
    value = paddle.rand([N, S, M, D]) * 0.01
    sampling_locations = paddle.rand([N, Lq, M, L, P, 2])
    attention_weights = paddle.rand([N, Lq, M, L, P]) + 1e-5
    attention_weights /= attention_weights.sum(axis=(-1), keepdim=True).sum(axis=(-2), keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value.numpy(), shapes.numpy(), sampling_locations.numpy(), attention_weights.numpy()).astype(np.float32)
    output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step).numpy()
    fwdok = np.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = np.max(np.abs(output_cuda - output_pytorch))
    max_rel_err = np.max(np.abs((output_cuda - output_pytorch) / output_pytorch))

    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = paddle.rand([N, S, M, channels]) * 0.01
    sampling_locations = paddle.rand([N, Lq, M, L, P, 2])
    attention_weights = paddle.rand([N, Lq, M, L, P]) + 1e-5
    attention_weights /= attention_weights.sum(axis=(-1), keepdim=True).sum(axis=(-2), keepdim=True)
    im2col_step = 2
    func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    gradok = gradcheck(func, (value.astype(np.float64), shapes.numpy(), level_start_index.numpy(), sampling_locations.astype(np.float64), attention_weights.astype(np.float64), im2col_step))

    print(f'* {gradok} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()

    for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
        check_gradient_numerical(channels, True, True, True)
