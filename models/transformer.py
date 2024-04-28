import paddle
import paddle.nn as nn

from models.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from models.transformer_decoder import TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Layer):
    def __init__(self, args, activation="relu"):
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        self.enc_layers = args.enc_layers
        self.dec_layers = args.dec_layers
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        if self.multiscale:
            # Reminder: To use multiscale SAM-DETR, you need to compile CUDA operators for Deformable Attention.
            from models.transformer_encoder import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
            self.num_feature_levels = 3  # Hard-coded multiscale parameters
            # Use Deformable Attention in Transformer Encoder for efficient computation of multiscale features
            encoder_layer = DeformableTransformerEncoderLayer(args, activation)
            self.encoder = DeformableTransformerEncoder(args, encoder_layer, self.enc_layers)
            self.level_embed = nn.Parameter(paddle.Tensor(self.num_feature_levels, self.d_model))
        else:
            encoder_layer = TransformerEncoderLayer(args, activation)
            self.encoder = TransformerEncoder(args, encoder_layer, self.enc_layers)

        decoder_layer = TransformerDecoderLayer(args, activation)
        self.decoder = TransformerDecoder(args, decoder_layer, self.dec_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.initializer.XavierUniform()(p)
        if self.multiscale:
            from models.ops.modules import MSDeformAttn
            for m in self.sublayers():
                if isinstance(m, MSDeformAttn):
                    m._reset_parameters()
            nn.initializer.Normal()(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = paddle.sum(~mask[:, :, 0], 1)
        valid_W = paddle.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.astype('float32') / H
        valid_ratio_w = valid_W.astype('float32') / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, query_embed, pos_embeds):
        if self.multiscale:
            return self.forward_multi_scale(srcs, masks, query_embed, pos_embeds)
        else:
            return self.forward_single_scale(srcs[0], masks[0], query_embed, pos_embeds[0])

    def forward_single_scale(self, src, mask, query_embed, pos):
        bs, c, memory_h, memory_w = src.shape

        if self.args.smca:
            grid_y, grid_x = paddle.meshgrid(paddle.arange(0, memory_h), paddle.arange(0, memory_w))
            grid = paddle.stack((grid_x, grid_y), 2).astype('float32').to(src.device)
            grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * self.nheads, 1)
        else:
            grid = None

        src = src.flatten(2).transpose(1, 2)  # flatten NxCxHxW to HWxNxC
        pos = pos.flatten(2).transpose(1, 2)  # flatten NxCxHxW to HWxNxC
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = paddle.zeros((self.num_queries, bs, c), device=query_embed.device)

        # encoder
        memory = self.encoder(src,
                              src_key_padding_mask=mask,
                              pos=pos)

        # decoder
        hs, references = self.decoder(tgt,
                                      memory,
                                      memory_key_padding_mask=mask,
                                      pos=pos,
                                      query_pos=query_embed,
                                      memory_h=memory_h,
                                      memory_w=memory_w,
                                      grid=grid)
        return hs, references

    def forward_multi_scale(self, srcs, masks, query_embed, pos_embeds):

        bs, c, h_16, w_16 = srcs[0].shape
        bs, c, h_32, w_32 = srcs[1].shape
        bs, c, h_64, w_64 = srcs[2].shape

        src_16 = srcs[0].flatten(2).transpose(1, 2)
        orig_pos_embed_16 = pos_embeds[0] + self.level_embed[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_16 = orig_pos_embed_16.flatten(2).transpose(1, 2)
        mask_16 = masks[0].flatten(1)

        src_32 = srcs[1].flatten(2).transpose(1, 2)
        orig_pos_embed_32 = pos_embeds[1] + self.level_embed[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_32 = orig_pos_embed_32.flatten(2).transpose(1, 2)
        mask_32 = masks[1].flatten(1)

        src_64 = srcs[2].flatten(2).transpose(1, 2)
        orig_pos_embed_64 = pos_embeds[2] + self.level_embed[2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_64 = orig_pos_embed_64.flatten(2).transpose(1, 2)
        mask_64 = masks[2].flatten(1)

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].unsqueeze(0).unsqueeze(0)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64', device=src_flatten.device)
        level_start_index = paddle.concat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten,
                              spatial_shapes,
                              level_start_index,
                              valid_ratios,
                              lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        tgt = paddle.zeros((self.num_queries, bs, self.d_model), device=query_embed.device)
        mem1 = memory[:, level_start_index[0]: level_start_index[1], :]
        mem2 = memory[:, level_start_index[1]: level_start_index[2], :]
        mem3 = memory[:, level_start_index[2]:, :]
        memory_flatten = []
        for m in [mem1, mem2, mem3]:
            memory_flatten.append(m.transpose(1, 0))
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        grid_y_16, grid_x_16 = paddle.meshgrid(paddle.arange(0, h_16), paddle.arange(0, w_16))
        grid_16 = paddle.stack((grid_x_16, grid_y_16), 2).astype('float32')
        grid_16 = grid_16.unsqueeze(0).transpose(0, 3).flatten(2).transpose(1, 0)
        grid_16 = grid_16.repeat(1, bs * 8, 1)

        grid_y_32, grid_x_32 = paddle.meshgrid(paddle.arange(0, h_32), paddle.arange(0, w_32))
        grid_32 = paddle.stack((grid_x_32, grid_y_32), 2).astype('float32')
        grid_32 = grid_32.unsqueeze(0).transpose(0, 3).flatten(2).transpose(1, 0)
        grid_32 = grid_32.repeat(1, bs * 8, 1)

        grid_y_64, grid_x_64 = paddle.meshgrid(paddle.arange(0, h_64), paddle.arange(0, w_64))
        grid_64 = paddle.stack((grid_x_64, grid_y_64), 2).astype('float32')
        grid_64 = grid_64.unsqueeze(0).transpose(0, 3).flatten(2).transpose(1, 0)
        grid_64 = grid_64.repeat(1, bs * 8, 1)

        # decoder
        hs, references = self.decoder(tgt,
                                      [memory_flatten[0], memory_flatten[1], memory_flatten[2]],
                                      memory_key_padding_mask=[mask_16, mask_32, mask_64],
                                      pos=[pos_embed_16, pos_embed_32, pos_embed_64],
                                      query_pos=query_embed,
                                      memory_h=[h_16, h_32, h_64],
                                      memory_w=[w_16, w_32, w_64],
                                      grid=[grid_16, grid_32, grid_64])

        return hs, references


def build_transformer(args):
    return Transformer(args, activation="relu")
