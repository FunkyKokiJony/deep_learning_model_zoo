"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jointnet.attention2d import MultiHeadAttention2D, conv_for_sequence


def expand(img_seq, expansion):
    if expansion < 2:
        return img_seq
    new_seq = []
    for idx in range(img_seq.shape[1]):
        new_seq.append(img_seq[:, idx:(idx + 1)].expand((-1, expansion, -1, -1, -1)).contiguous())

    return torch.cat(new_seq, dim=1)

class EncoderLayer2D(nn.Module):
    def __init__(self, h, w, channels, expansion
                 , q_out_channels, q_kernel, q_stride, q_padding
                 , k_out_channels, k_kernel, k_stride, k_padding
                 , v_out_channels, v_kernel, v_stride, v_padding
                 , transform_kernel, transform_stride, transform_padding
                 , ff_out_channels, ff_kernel, ff_stride, ff_padding):
        super().__init__()
        # input size is (B, S, C, H, W)
        self.h_0, self.w_0 = h, w
        self.expansion = expansion
        self.mha2d = MultiHeadAttention2D(q_in_channels=channels
                                            , q_out_channels=q_out_channels
                                            , q_kernel=q_kernel
                                            , q_stride=q_stride
                                            , q_padding=q_padding
                                            , k_in_channels=channels
                                            , k_out_channels=k_out_channels
                                            , k_kernel=k_kernel
                                            , k_stride=k_stride
                                            , k_padding=k_padding
                                            , v_in_channels=channels
                                            , v_out_channels=v_out_channels
                                            , v_kernel=v_kernel
                                            , v_stride=v_stride
                                            , v_padding=v_padding
                                            , transform_kernel=transform_kernel
                                            , transform_stride=transform_stride
                                            , transform_padding=transform_padding)
        self.h_1, self.w_1 = (self.h_0 - v_kernel + 2 * v_padding) // v_stride + 1\
            , (self.w_0 - v_kernel + 2 * v_padding) // v_stride + 1
        self.ln_1 = nn.LayerNorm([v_out_channels, self.h_1, self.w_1])
        self.downsample_1 = None
        if self.h_1 != self.h_0 or self.w_1 != self.w_0 or channels != v_out_channels:
            self.downsample_1 = nn.Conv2d(channels, v_out_channels, kernel_size=v_kernel
                                          , stride=v_stride, padding=v_padding)

        self.conv = nn.Conv2d(v_out_channels, ff_out_channels, kernel_size=ff_kernel
                              , stride=ff_stride, padding=ff_padding)
        self.h_2, self.w_2 = (self.h_1 - ff_kernel + 2 * ff_padding) // ff_stride + 1\
            , (self.w_1 - ff_kernel + 2 * ff_padding) // ff_stride + 1

        self.ln_2 = nn.LayerNorm([ff_out_channels, self.h_2, self.w_2])
        self.downsample_2 = None
        if self.h_2 != self.h_1 or self.w_2 != self.w_1 or v_out_channels != ff_out_channels:
            self.downsample_2 = nn.Conv2d(v_out_channels, ff_out_channels, kernel_size=ff_kernel
                                          , stride=ff_stride, padding=ff_padding)

        self.out_channels = ff_out_channels
        self.out_h, self.out_w = self.h_2, self.w_2

    def forward(self, img_seq):
        # input size is (B, S, C, H, W)
        img_seq = expand(img_seq, self.expansion)

        temp = self.mha2d(img_seq, img_seq, img_seq)
        if self.downsample_1 is not None:
            img_seq = F.relu(conv_for_sequence(img_seq, self.downsample_1) + temp, inplace=True)
        else:
            img_seq = F.relu(img_seq + temp, inplace=True)
        img_seq = self.ln_1(img_seq)

        temp = F.relu(conv_for_sequence(img_seq, self.conv), inplace=True)
        if self.downsample_2 is not None:
            img_seq = F.relu(conv_for_sequence(img_seq, self.downsample_2) + temp, inplace=True)
        else:
            img_seq = F.relu(img_seq + temp, inplace=True)
        img_seq = self.ln_2(img_seq)

        return img_seq

class DecoderLayer2D(nn.Module):
    def __init__(self, h, w, channels, expansion
                 , q1_out_channels, q1_kernel, q1_stride, q1_padding
                 , k1_out_channels, k1_kernel, k1_stride, k1_padding
                 , v1_out_channels, v1_kernel, v1_stride, v1_padding
                 , transform1_kernel, transform1_stride, transform1_padding
                 , kv_h, kv_w, kv_channels
                 , q2_out_channels, q2_kernel, q2_stride, q2_padding
                 , k2_out_channels, k2_kernel, k2_stride, k2_padding
                 , v2_out_channels, v2_kernel, v2_stride, v2_padding
                 , transform2_kernel, transform2_stride, transform2_padding
                 , ff_out_channels, ff_kernel, ff_stride, ff_padding
                 , target_h, target_w, upsample_out_channels, upsample_kernel, upsample_stride, upsample_padding):
        super().__init__()
        # input size is (B, S, C, H, W)
        self.h_0, self.w_0 = h, w
        self.expansion = expansion
        self.target_h, self.target_w = target_h, target_w
        self.mha2d_1 = MultiHeadAttention2D(q_in_channels=channels
                                            , q_out_channels=q1_out_channels
                                            , q_kernel=q1_kernel
                                            , q_stride=q1_stride
                                            , q_padding=q1_padding
                                            , k_in_channels=channels
                                            , k_out_channels=k1_out_channels
                                            , k_kernel=k1_kernel
                                            , k_stride=k1_stride
                                            , k_padding=k1_padding
                                            , v_in_channels=channels
                                            , v_out_channels=v1_out_channels
                                            , v_kernel=v1_kernel
                                            , v_stride=v1_stride
                                            , v_padding=v1_padding
                                            , transform_kernel=transform1_kernel
                                            , transform_stride=transform1_stride
                                            , transform_padding=transform1_padding)
        self.h_1, self.w_1 = (self.h_0 - v1_kernel + 2 * v1_padding) // v1_stride + 1\
            , (self.w_0 - v1_kernel + 2 * v1_padding) // v1_stride + 1
        self.ln_1 = nn.LayerNorm([v1_out_channels, self.h_1, self.w_1])
        self.downsample_1 = None
        if self.h_1 != self.h_0 or self.w_1 != self.w_0 or channels != v1_out_channels:
            self.downsample_1 = nn.Conv2d(channels, v1_out_channels, kernel_size=v1_kernel
                                          , stride=v1_stride, padding=v1_padding)

        self.mha2d_2 = MultiHeadAttention2D(q_in_channels=v1_out_channels
                                            , q_out_channels=q2_out_channels
                                            , q_kernel=q2_kernel
                                            , q_stride=q2_stride
                                            , q_padding=q2_padding
                                            , k_in_channels=kv_channels
                                            , k_out_channels=k2_out_channels
                                            , k_kernel=k2_kernel
                                            , k_stride=k2_stride
                                            , k_padding=k2_padding
                                            , v_in_channels=kv_channels
                                            , v_out_channels=v2_out_channels
                                            , v_kernel=v2_kernel
                                            , v_stride=v2_stride
                                            , v_padding=v2_padding
                                            , transform_kernel=transform2_kernel
                                            , transform_stride=transform2_stride
                                            , transform_padding=transform2_padding)

        self.h_2, self.w_2 = (kv_h - v2_kernel + 2 * v2_padding) // v2_stride + 1\
            , (kv_w - v2_kernel + 2 * v2_padding) // v2_stride + 1
        self.ln_2 = nn.LayerNorm([v2_out_channels, self.h_2, self.w_2])
        self.downsample_2 = None
        if self.h_2 != self.h_1 or self.w_2 != self.w_1 or v1_out_channels != v2_out_channels:
            self.downsample_2 = nn.Conv2d(v1_out_channels, v2_out_channels, kernel_size=v2_kernel
                                          , stride=v2_stride, padding=v2_padding)

        self.conv = nn.Conv2d(v2_out_channels, ff_out_channels, kernel_size=ff_kernel
                              , stride=ff_stride, padding=ff_padding)
        self.h_3, self.w_3 = (self.h_2 - ff_kernel + 2 * ff_padding) // ff_stride + 1\
            , (self.w_2 - ff_kernel + 2 * ff_padding) // ff_stride + 1

        self.ln_3 = nn.LayerNorm([ff_out_channels, self.h_3, self.w_3])
        self.downsample_3 = None
        if self.h_3 != self.h_2 or self.w_3 != self.w_2 or v2_out_channels != ff_out_channels:
            self.downsample_3 = nn.Conv2d(v2_out_channels, ff_out_channels, kernel_size=ff_kernel
                                          , stride=ff_stride, padding=ff_padding)

        self.out_channels = ff_out_channels
        self.out_h, self.out_w = self.h_3, self.w_3

        self.upsample_conv = None
        if target_h is not None and target_w is not None and (target_h != self.h_3 or target_w != self.w_3):
            self.upsample_conv = nn.Conv2d(ff_out_channels, upsample_out_channels
                                        , kernel_size=upsample_kernel, stride=upsample_stride
                                        , padding=upsample_padding)
            self.out_h, self.out_w = self.target_h, self.target_w
            self.out_channels = upsample_out_channels


    def forward(self, img_seq, kv):
        # input size is (B, S, C, H, W)
        img_seq = expand(img_seq, self.expansion)

        temp = self.mha2d_1(img_seq, img_seq, img_seq)
        if self.downsample_1 is not None:
            img_seq = F.relu(conv_for_sequence(img_seq, self.downsample_1) + temp, inplace=True)
        else:
            img_seq = F.relu(img_seq + temp, inplace=True)
        img_seq = self.ln_1(img_seq)

        temp = self.mha2d_2(img_seq, kv, kv)
        if self.downsample_2 is not None:
            img_seq = F.relu(conv_for_sequence(img_seq, self.downsample_2) + temp, inplace=True)
        else:
            img_seq = F.relu(img_seq + temp, inplace=True)
        img_seq = self.ln_2(img_seq)

        temp = F.relu(conv_for_sequence(img_seq, self.conv), inplace=True)
        if self.downsample_3 is not None:
            img_seq = F.relu(conv_for_sequence(img_seq, self.downsample_3) + temp, inplace=True)
        else:
            img_seq = F.relu(img_seq + temp, inplace=True)
        img_seq = self.ln_3(img_seq)

        if self.upsample_conv is not None:
            _shape = img_seq.shape
            img_seq = F.interpolate(
                img_seq.view(-1, _shape[2], _shape[3], _shape[4])
                , (self.out_h, self.out_w)
            ).view(_shape[0], -1, _shape[2], self.out_h, self.out_w).contiguous()
            img_seq = F.relu(conv_for_sequence(img_seq, self.upsample_conv), inplace=True)

        return img_seq


class Encoder2D(nn.Module):
    def __init__(self, h, w, channels, expansions):
        super().__init__()
        # assuming the input image size is (B, 3, 256, 256)
        # input size is (B, 2, 3, 256, 256)
        self.expansions = expansions
        self.encoder_layer_1 = EncoderLayer2D(h=h
                                            , w=w
                                            , channels=channels
                                            , expansion=expansions[0]
                                            , q_out_channels=32
                                            , q_kernel=7
                                            , q_stride=2
                                            , q_padding=0
                                            , k_out_channels=32
                                            , k_kernel=7
                                            , k_stride=2
                                            , k_padding=0
                                            , v_out_channels=32
                                            , v_kernel=7
                                            , v_stride=2
                                            , v_padding=0
                                            , transform_kernel=1
                                            , transform_stride=1
                                            , transform_padding=0
                                            , ff_out_channels=32
                                            , ff_kernel=3
                                            , ff_stride=1
                                            , ff_padding=1)
        # output size is (B, 2, 32, 125, 125)

        # input size is (B, 2, 32, 125, 125)
        self.encoder_layer_2 = EncoderLayer2D(h=self.encoder_layer_1.out_h
                                            , w=self.encoder_layer_1.out_w
                                            , channels=self.encoder_layer_1.out_channels
                                            , expansion=expansions[1]
                                            , q_out_channels=32
                                            , q_kernel=7
                                            , q_stride=2
                                            , q_padding=0
                                            , k_out_channels=32
                                            , k_kernel=7
                                            , k_stride=2
                                            , k_padding=0
                                            , v_out_channels=32
                                            , v_kernel=7
                                            , v_stride=2
                                            , v_padding=0
                                            , transform_kernel=1
                                            , transform_stride=1
                                            , transform_padding=0
                                            , ff_out_channels=32
                                            , ff_kernel=3
                                            , ff_stride=1
                                            , ff_padding=1)
        # output size is (B, 4, 32, 60, 60)

        # input size is (B, 4, 32, 60, 60)
        self.encoder_layer_3 = EncoderLayer2D(h=self.encoder_layer_2.out_h
                                              , w=self.encoder_layer_2.out_w
                                              , channels=self.encoder_layer_2.out_channels
                                              , expansion=expansions[2]
                                              , q_out_channels=64
                                              , q_kernel=7
                                              , q_stride=2
                                              , q_padding=0
                                              , k_out_channels=64
                                              , k_kernel=7
                                              , k_stride=2
                                              , k_padding=0
                                              , v_out_channels=64
                                              , v_kernel=7
                                              , v_stride=2
                                              , v_padding=0
                                              , transform_kernel=1
                                              , transform_stride=1
                                              , transform_padding=0
                                              , ff_out_channels=64
                                              , ff_kernel=3
                                              , ff_stride=1
                                              , ff_padding=1)
        # output size is (B, 8, 64, 27, 27)

    def forward(self, img_seq):
        embeddings = []

        img_seq = self.encoder_layer_1(img_seq)
        embeddings.append(img_seq)

        img_seq = self.encoder_layer_2(img_seq)
        embeddings.append(img_seq)

        img_seq = self.encoder_layer_3(img_seq)
        embeddings.append(img_seq)

        return embeddings

class Decoder2D(nn.Module):
    def __init__(self, h, w, channels, kvs_h, kvs_w, kvs_channels):
        super().__init__()
        # use upsampling + conv instead of deconv
        # the upsampling + conv is performed in the layer after multi-head attention
        # assuming the input scope size is (B, 1, 1, 256, 256)
        # input size is (B, 1, 1, 256, 256)
        self.encoder = Encoder2D(h, w, channels, [1, 1, 1])
        # output size is (B, 1, 64, 27, 27)

        # input q size is (B, 1, 64, 27, 27)
        # input kv size is (B, 8, 64, 27, 27)
        self.decoder_layer_1 = DecoderLayer2D(h=self.encoder.encoder_layer_3.out_h
                                              , w=self.encoder.encoder_layer_3.out_w
                                              , channels=self.encoder.encoder_layer_3.out_channels
                                              , expansion=1
                                              , q1_out_channels=64
                                              , q1_kernel=7
                                              , q1_stride=1
                                              , q1_padding=3
                                              , k1_out_channels=64
                                              , k1_kernel=7
                                              , k1_stride=1
                                              , k1_padding=3
                                              , v1_out_channels=64
                                              , v1_kernel=7
                                              , v1_stride=1
                                              , v1_padding=3
                                              , transform1_kernel=1
                                              , transform1_stride=1
                                              , transform1_padding=0
                                              , kv_h=kvs_h[-1]
                                              , kv_w=kvs_w[-1]
                                              , kv_channels=kvs_channels[-1]
                                              , q2_out_channels=64
                                              , q2_kernel=7
                                              , q2_stride=1
                                              , q2_padding=3
                                              , k2_out_channels=64
                                              , k2_kernel=7
                                              , k2_stride=1
                                              , k2_padding=3
                                              , v2_out_channels=64
                                              , v2_kernel=7
                                              , v2_stride=1
                                              , v2_padding=3
                                              , transform2_kernel=1
                                              , transform2_stride=1
                                              , transform2_padding=0
                                              , ff_out_channels=64
                                              , ff_kernel=3
                                              , ff_stride=1
                                              , ff_padding=1
                                              , target_h=kvs_h[-2]
                                              , target_w=kvs_w[-2]
                                              , upsample_out_channels=kvs_channels[-2]
                                              , upsample_kernel=3
                                              , upsample_stride=1
                                              , upsample_padding=1)
        # output size is (B, 1, 32, 60, 60)

        # input size is (B, 1, 32, 60, 60)
        # input kv size is (B, 4, 32, 60, 60)
        self.decoder_layer_2 = DecoderLayer2D(h=self.decoder_layer_1.out_h
                                              , w=self.decoder_layer_1.out_w
                                              , channels=self.decoder_layer_1.out_channels
                                              , expansion=1
                                              , q1_out_channels=32
                                              , q1_kernel=7
                                              , q1_stride=1
                                              , q1_padding=3
                                              , k1_out_channels=32
                                              , k1_kernel=7
                                              , k1_stride=1
                                              , k1_padding=3
                                              , v1_out_channels=32
                                              , v1_kernel=7
                                              , v1_stride=1
                                              , v1_padding=3
                                              , transform1_kernel=1
                                              , transform1_stride=1
                                              , transform1_padding=0
                                              , kv_h=kvs_h[-2]
                                              , kv_w=kvs_w[-2]
                                              , kv_channels=kvs_channels[-2]
                                              , q2_out_channels=32
                                              , q2_kernel=7
                                              , q2_stride=1
                                              , q2_padding=3
                                              , k2_out_channels=32
                                              , k2_kernel=7
                                              , k2_stride=1
                                              , k2_padding=3
                                              , v2_out_channels=32
                                              , v2_kernel=7
                                              , v2_stride=1
                                              , v2_padding=3
                                              , transform2_kernel=1
                                              , transform2_stride=1
                                              , transform2_padding=0
                                              , ff_out_channels=32
                                              , ff_kernel=3
                                              , ff_stride=1
                                              , ff_padding=1
                                              , target_h=kvs_h[-3]
                                              , target_w=kvs_w[-3]
                                              , upsample_out_channels=kvs_channels[-3]
                                              , upsample_kernel=3
                                              , upsample_stride=1
                                              , upsample_padding=1)
        # output size is (B, 1, 32, 125, 125)

        # input size is (B, 1, 32, 125, 125)
        # input kv size is (B, 2, 32, 125, 125)
        self.decoder_layer_3 = DecoderLayer2D(h=self.decoder_layer_2.out_h
                                              , w=self.decoder_layer_2.out_w
                                              , channels=self.decoder_layer_2.out_channels
                                              , expansion=1
                                              , q1_out_channels=32
                                              , q1_kernel=7
                                              , q1_stride=1
                                              , q1_padding=3
                                              , k1_out_channels=32
                                              , k1_kernel=7
                                              , k1_stride=1
                                              , k1_padding=3
                                              , v1_out_channels=32
                                              , v1_kernel=7
                                              , v1_stride=1
                                              , v1_padding=3
                                              , transform1_kernel=1
                                              , transform1_stride=1
                                              , transform1_padding=0
                                              , kv_h=kvs_h[-3]
                                              , kv_w=kvs_w[-3]
                                              , kv_channels=kvs_channels[-3]
                                              , q2_out_channels=32
                                              , q2_kernel=7
                                              , q2_stride=1
                                              , q2_padding=3
                                              , k2_out_channels=32
                                              , k2_kernel=7
                                              , k2_stride=1
                                              , k2_padding=3
                                              , v2_out_channels=32
                                              , v2_kernel=7
                                              , v2_stride=1
                                              , v2_padding=3
                                              , transform2_kernel=1
                                              , transform2_stride=1
                                              , transform2_padding=0
                                              , ff_out_channels=16
                                              , ff_kernel=3
                                              , ff_stride=1
                                              , ff_padding=1
                                              , target_h=h
                                              , target_w=w
                                              , upsample_out_channels=8
                                              , upsample_kernel=3
                                              , upsample_stride=1
                                              , upsample_padding=1)
        # output size is (B, 1, 8, 256, 256)

    def forward(self, img_seq, embeddings):

        img_seq = self.encoder(img_seq)[-1]

        img_seq = self.decoder_layer_1(img_seq, embeddings[-1])

        img_seq = self.decoder_layer_2(img_seq, embeddings[-2])

        img_seq = self.decoder_layer_3(img_seq, embeddings[-3])

        return img_seq


class AirTransformer2D(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.encoder = Encoder2D(256, 256, 3, [2, 2, 2])
        self.decoder = Decoder2D(256, 256, 1, [125, 60, 27], [125, 60, 27], [32, 32, 64])
        self.conv = nn.Conv2d(12, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, imgs, scopes):
        img_seq = torch.unsqueeze(imgs, 1)
        scope_seq = torch.unsqueeze(scopes, 1)

        embeddings = self.encoder(img_seq)

        mask_seq = []

        for idx in range(self.steps):
            masks = torch.sigmoid(self.conv(torch.cat((imgs, scopes, self.decoder(scope_seq, embeddings).squeeze(dim=1)), dim=1)))
            scopes = (1 - masks) * scopes
            scope_seq = torch.unsqueeze(scopes, 1)
            mask_seq.append(masks)

        return torch.stack(mask_seq, dim=1)




