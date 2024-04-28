import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

# PaddlePaddle version of _LinearWithBias
class _LinearWithBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias_attr=True)

# PaddlePaddle version of MultiheadAttention
class MultiheadAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0., add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.out_proj = _LinearWithBias(self.vdim, self.vdim)
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        self.out_proj.bias.set_value(0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, gaussian=None):
        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim

        head_dim = self.embed_dim // self.num_heads
        v_head_dim = self.vdim // self.num_heads
        scaling = float(head_dim) ** -0.5

        q = query * scaling
        k = key
        v = value

        if attn_mask is not None:
            attn_mask = paddle.unsqueeze(attn_mask, axis=0)
            if attn_mask.shape != [1, query.shape[0], key.shape[0]]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')

        if key_padding_mask is not None and key_padding_mask.dtype == paddle.bool:
            key_padding_mask = paddle.cast(key_padding_mask, dtype=paddle.float32)

        if self.add_zero_attn:
            src_len = k.shape[1]
            src_len += 1
            k = paddle.concat([k, paddle.zeros((k.shape[0], 1) + k.shape[2:], dtype=k.dtype)], axis=1)
            v = paddle.concat([v, paddle.zeros((v.shape[0], 1) + v.shape[2:], dtype=v.dtype)], axis=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, [[0, 0], [0, 1], [0, 1]])
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, [[0, 0], [0, 1]])

        q = paddle.transpose(q, perm=[1, 0, 2])
        k = paddle.transpose(k, perm=[1, 0, 2])
        v = paddle.transpose(v, perm=[1, 0, 2])

        attn_output_weights = paddle.matmul(q, k.transpose(perm=[0, 2, 1]))
        assert attn_output_weights.shape == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = paddle.reshape(attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = paddle.where(paddle.unsqueeze(key_padding_mask, axis=[1, 2]), float('-inf'), attn_output_weights)
            attn_output_weights = paddle.reshape(attn_output_weights, [bsz * self.num_heads, tgt_len, src_len])

        if gaussian is not None:
            attn_output_weights += paddle.transpose(gaussian[0], perm=[2, 0, 1])
        
        attn_output_weights = F.softmax(attn_output_weights, axis=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_output_weights, v)
        attn_output = paddle.transpose(attn_output, perm=[1, 0, 2])
        attn_output = paddle.reshape(attn_output, [tgt_len, bsz, self.vdim])
        attn_output = paddle.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            attn_output_weights = paddle.reshape(attn_output_weights, [bsz, self.num_heads, tgt_len, src_len])
            attn_output_weights = paddle.sum(attn_output_weights, axis=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, None

def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 bias_k=None,
                                 bias_v=None,
                                 add_zero_attn=False,
                                 dropout_p=0.,
                                 out_proj_weight=None,
                                 out_proj_bias=None,
                                 training=True,
                                 key_padding_mask=None,
                                 need_weights=True,
                                 attn_mask=None,
                                 use_separate_proj_weight=False,
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None,
                                 out_dim=None,
                                 gaussian=None):
    tgt_len, bsz, embed_dim = query.shape
    assert embed_dim == embed_dim_to_check
    assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]

    head_dim = embed_dim // num_heads
    v_head_dim = out_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q = query * scaling
    k = key
    v = value

    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            attn_mask = paddle.cast(attn_mask, dtype=paddle.float32)
        if attn_mask.ndim == 2:
            attn_mask = paddle.unsqueeze(attn_mask, axis=0)
            if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.ndim == 3:
            if list(attn_mask.shape) != [bsz * num_heads, query.shape[0], key.shape[0]]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.ndim))

    if key_padding_mask is not None and key_padding_mask.dtype == paddle.bool:
        key_padding_mask = paddle.cast(key_padding_mask, dtype=paddle.float32)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = paddle.concat([k, bias_k.tile((1, bsz, 1))], axis=1)
            v = paddle.concat([v, bias_v.tile((1, bsz, 1))], axis=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, [0, 0, 0, 1])
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, [0, 1])
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.transpose((1, 0, 2))
    if k is not None:
        k = k.transpose((1, 0, 2))
    if v is not None:
        v = v.transpose((1, 0, 2))

    if static_k is not None:
        assert static_k.shape[0] == bsz * num_heads
        assert static_k.shape[2] == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.shape[0] == bsz * num_heads
        assert static_v.shape[2] == v_head_dim
        v = static_v

    src_len = k.shape[1]

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len

    if add_zero_attn:
        src_len += 1
        k = paddle.concat([k, paddle.zeros((k.shape[0], 1) + k.shape[2:], dtype=k.dtype)], axis=1)
        v = paddle.concat([v, paddle.zeros((v.shape[0], 1) + v.shape[2:], dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, [0, 0, 0, 1])
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, [0, 1])

    attn_output_weights = paddle.matmul(q, k.transpose((0, 2, 1)))
    assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.reshape([bsz, num_heads, tgt_len, src_len])
        attn_output_weights = attn_output_weights.masked_fill(paddle.unsqueeze(key_padding_mask, axis=[1, 2]), float('-inf'))
        attn_output_weights = attn_output_weights.reshape([bsz * num_heads, tgt_len, src_len])

    if gaussian is not None:
        attn_output_weights += paddle.transpose(gaussian[0], perm=[2, 0, 1])
    
    attn_output_weights = F.softmax(attn_output_weights, axis=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = paddle.matmul(attn_output_weights, v)
    attn_output = attn_output.transpose((1, 0, 2)).reshape([tgt_len, bsz, out_dim])
    attn_output = paddle.nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.reshape([bsz, num_heads, tgt_len, src_len])
        attn_output_weights = paddle.sum(attn_output_weights, axis=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None
