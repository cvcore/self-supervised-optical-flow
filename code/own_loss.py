import torch
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample
from util import save_image, flow2rgb
import time
import numpy as np
from imageio import imwrite

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def charbonnier_loss(input, alpha, beta):
    eps = 0.001 # from reference implementation
    sq = input*input*beta*beta + torch.ones(input.shape).to(device) * eps*eps
    return torch.sum(torch.pow(sq, alpha))


def photometric_loss(im1, im2, flow, negative_flow = False):
    # upscaling in case the height does not match. Assumes image ratio is correct
    if im1.shape[2] != flow.shape[2]:
        m = Upsample(scale_factor=im1.shape[2]/flow.shape[2], mode='bilinear').to(device)
        flow = m(flow)

    # copied from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py

    if negative_flow:
        x = im1
    else:
        x = im2

    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)

    if negative_flow:
        vgrid = grid - flow
    else:
        vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if negative_flow:
        output = F.grid_sample(im1, vgrid)
    else:
        output = F.grid_sample(im2, vgrid)

    # for debug purpose
    # save_image(im1[0], 'im1.png')
    # save_image(im2[0], 'im2.png')
    # save_image((output)[0], 'diff.png')
    # return

    # apply charbonnier loss
    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    return charbonnier_loss(output-im1, 2, 1)


def smoothness_loss(flow):
    diff_y = flow[:, :, :, :-1] - flow[:, :, :, 1:]
    diff_x = flow[:, :, :-1, :] - flow[:, :, 1:, :]

    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    return 100*charbonnier_loss(diff_x, 2, 1) + \
           100*charbonnier_loss(diff_y, 2, 1)






