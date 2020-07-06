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


def photometric_loss(im1, im2, flow, config):
    negative_flow = config['negative_flow']
    pl_exp = config['pl_exp']

    # upscaling in case the height does not match. Assumes image ratio is correct
    if im1.shape[2] != flow.shape[2]:
        flow = F.interpolate(input=flow, scale_factor=im1.shape[2]/flow.shape[2], mode='bilinear').to(device)

    # adapted from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
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
    if negative_flow:
        return charbonnier_loss(output - im2, pl_exp, 1)
    else:
        return charbonnier_loss(output - im1, pl_exp, 1)


def smoothness_loss(flow, config):
    sl_weight = config['sl_weight']
    sl_exp = config['sl_exp']

    diff_x = flow[:, :, :-1, :] - flow[:, :, 1:, :]
    diff_y = flow[:, :, :, :-1] - flow[:, :, :, 1:]

    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    return sl_weight*charbonnier_loss(diff_x, sl_exp, 1) + \
           sl_weight*charbonnier_loss(diff_y, sl_exp, 1)

def weighted_smoothness_loss(im1, im2, flow, config):
    # calculates |grad U_x| * exp(-|grad I_x|) +
    #            |grad U_y| * exp(-|grad I_y|) +
    #            |grad V_x| * exp(-|grad I_x|) +
    #            |grad V_y| * exp(-|grad I_y|)

    sl_weight = config['sl_weight']
    negative_flow = config['negative_flow']

    # todo: no idea which image to take...
    if negative_flow:
        image = im1
    else:
        image = im2

    # todo: no idea if downsampling or upsampling is better...
    if image.shape[2] != flow.shape[2]:
        image = F.interpolate(input=image, scale_factor=flow.shape[2]/im1.shape[2], mode='bilinear').to(device)

    diff_flow_x = abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
    diff_flow_y = abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])

    diff_img_x = image[:, :, :-1, :] - image[:, :, 1:, :]
    diff_img_y = image[:, :, :, :-1] - image[:, :, :, 1:]

    exp_x = torch.exp(-torch.sum(abs(diff_img_x), dim=1)).unsqueeze(1).expand(-1,2,-1,-1)
    exp_y = torch.exp(-torch.sum(abs(diff_img_y), dim=1)).unsqueeze(1).expand(-1,2,-1,-1)

    return sl_weight*torch.sum(diff_flow_x * exp_x) + \
           sl_weight*torch.sum(diff_flow_y * exp_y)











