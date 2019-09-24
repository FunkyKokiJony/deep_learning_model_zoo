"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jointnet.air_transformer2d import AirTransformer2D
from models.jointnet.vqvae import VQVAE


class MONet(nn.Module):
    def __init__(self, steps=8):
        super().__init__()
        self.steps = steps
        self.transformer = AirTransformer2D(self.steps)
        self.vqvae = VQVAE(in_channel=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, imgs, scopes):
        #Assuming the input layout is (N, C, H, W)
        attention_seq = self.transformer(imgs, scopes)
        imgs_seq = torch.unsqueeze(imgs, dim=1)
        imgs_seq = imgs_seq.expand((-1, self.steps, -1, -1, -1))
        inputs = torch.cat((attention_seq, imgs_seq), dim=2)
        inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        # The output layout is (N * S, 1 + C, H, W)
        outputs, latent_loss = self.vqvae(inputs)

        return attention_seq\
            , outputs[:, 0, :, :].reshape(imgs.shape[0], -1, 1, imgs.shape[2], imgs.shape[3])\
            , outputs[:, 1:, :, :].reshape(imgs.shape[0], -1, 3, imgs.shape[2], imgs.shape[3])\
            , latent_loss

    def calculate_loss(self, outputs, imgs, latent_loss_weight, mask_kl_weight):
        #layout is (N, S, C, H, W)
        attention_seq, mask_seq, recon_imgs_seq, latent_loss = outputs

        mask_seq = self.softmax(mask_seq)
        mask_kl_loss = F.kl_div(mask_seq.log(), attention_seq, reduction = 'sum') / imgs.shape[0] / imgs.shape[2] / imgs.shape[3]

        latent_loss = latent_loss.mean()

        mask_seq = mask_seq.expand((-1, -1, 3, -1, -1))
        recon = torch.mul(mask_seq, recon_imgs_seq)

        attention_seq = attention_seq.expand((-1, -1, 3, -1, -1))
        imgs_seq = torch.unsqueeze(imgs, dim=1)
        imgs_seq = imgs_seq.expand((-1, self.steps, -1, -1, -1))
        target = torch.mul(attention_seq, imgs_seq)

        recon_loss = F.mse_loss(recon, target)

        return recon_loss + latent_loss_weight * latent_loss + mask_kl_weight * mask_kl_loss

