import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import time
from util import save_image
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def image_warp(image, flow, with_mask=False):
    if image.shape[2] != flow.shape[2]:
        scale_factor = image.shape[2]/flow.shape[2]
        flow = F.interpolate(input=flow, scale_factor=image.shape[2]/flow.shape[2], mode='bilinear')
        flow *= scale_factor

    B, C, H, W = image.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)
    vgrid = torch.autograd.Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(image, vgrid)

    if with_mask:
        mask = torch.autograd.Variable(torch.ones(flow.size())).cuda()
        mask = F.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        mask = torch.sum(mask, dim=1)
        mask[mask > 0] = 1

        return output, mask
    else:
        return output


def length_sq(mat):
    return torch.sum(torch.pow(mat, 2), dim=1)

def all_losses(im1, im2, flow_fw, flow_bw, config, weight):
    sec_sl_weight = config['sec_sl_weight']
    sl_weight = config['sl_weight']
    wsl_weight = config['wsl_weight']
    fb_weight = config['fb_weight']
    pl_weight = config['pl_weight']
    sec_sl_exp = config['sec_sl_exp']
    sl_exp = config['sl_exp']
    fb_exp = config['fb_exp']
    pl_exp = config['pl_exp']

    scale_factor_fw = im1.shape[2] / flow_fw.shape[2]
    scale_factor_bw = im2.shape[2] / flow_bw.shape[2]
    flow_fw = F.interpolate(input=flow_fw, scale_factor=scale_factor_fw, mode='bilinear')
    flow_bw = F.interpolate(input=flow_bw, scale_factor=scale_factor_bw, mode='bilinear')
    flow_fw = flow_fw * scale_factor_fw
    flow_bw = flow_bw * scale_factor_bw

    im2_warped, mask_fw = image_warp(im2, flow_fw, with_mask=True)
    im1_warped, mask_bw = image_warp(im1, flow_bw, with_mask=True)

    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw = 0.01 * mag_sq_fw + 0.5
    occ_thresh_bw = 0.01 * mag_sq_bw + 0.5

    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()
    mask_fw *= (1 - fb_occ_fw)
    mask_bw *= (1 - fb_occ_bw)

    diff_flow_fw_y = flow_fw[:, :, 1:, :] - flow_fw[:, :, :-1, :]
    diff_flow_fw_x = flow_fw[:, :, :, 1:] - flow_fw[:, :, :, :-1]
    diff_flow_bw_y = flow_bw[:, :, 1:, :] - flow_bw[:, :, :-1, :]
    diff_flow_bw_x = flow_bw[:, :, :, 1:] - flow_bw[:, :, :, :-1]

    sec_diff_flow_fw_y = diff_flow_fw_y[:, :, 1:, :] - diff_flow_fw_y[:, :, :-1, :]
    sec_diff_flow_fw_x = diff_flow_fw_x[:, :, :, 1:] - diff_flow_fw_x[:, :, :, :-1]
    sec_diff_flow_bw_y = diff_flow_bw_y[:, :, 1:, :] - diff_flow_bw_y[:, :, :-1, :]
    sec_diff_flow_bw_x = diff_flow_bw_x[:, :, :, 1:] - diff_flow_bw_x[:, :, :, :-1]

    diff_im1_y = abs(im1[:, :, 1:, :] - im1[:, :, :-1, :])
    diff_im1_x = abs(im1[:, :, :, 1:] - im1[:, :, :, :-1])
    diff_im2_y = abs(im2[:, :, 1:, :] - im2[:, :, :-1, :])
    diff_im2_x = abs(im2[:, :, :, 1:] - im2[:, :, :, :-1])

    exp_im1_y = torch.exp(-torch.mean(diff_im1_y, dim=1, keepdim=True)).expand(-1,2,-1,-1)
    exp_im1_x = torch.exp(-torch.mean(diff_im1_x, dim=1, keepdim=True)).expand(-1,2,-1,-1)
    exp_im2_y = torch.exp(-torch.mean(diff_im2_y, dim=1, keepdim=True)).expand(-1,2,-1,-1)
    exp_im2_x = torch.exp(-torch.mean(diff_im2_x, dim=1, keepdim=True)).expand(-1,2,-1,-1)

    mask_fw = mask_fw.unsqueeze(1)
    mask_bw = mask_bw.unsqueeze(1)
    mask_fw_2 = mask_fw.expand(-1, 2, -1, -1)
    mask_fw_3 = mask_fw.expand(-1, 3, -1, -1)
    mask_bw_2 = mask_bw.expand(-1, 2, -1, -1)
    mask_bw_3 = mask_bw.expand(-1, 3, -1, -1)
    mask_fw_2_crop_1_y = mask_fw_2[:, :, 1:, :]
    mask_fw_2_crop_1_x = mask_fw_2[:, :, :, 1:]
    mask_bw_2_crop_1_y = mask_bw_2[:, :, 1:, :]
    mask_bw_2_crop_1_x = mask_bw_2[:, :, :, 1:]
    mask_fw_2_crop_2_y = mask_fw_2_crop_1_y[:, :, :-1, :]
    mask_fw_2_crop_2_x = mask_fw_2_crop_1_x[:, :, :, :-1]
    mask_bw_2_crop_2_y = mask_bw_2_crop_1_y[:, :, :-1, :]
    mask_bw_2_crop_2_x = mask_bw_2_crop_1_x[:, :, :, :-1]

    # for debug purpose
    # save_image(im1[0], 'im1.png')
    # save_image(im2[0], 'im2.png')
    # save_image(flow_fw[0], 'flow_fw.png')
    # save_image(flow_bw[0], 'flow_bw.png')
    # save_image(im2_warped[0], 'im2_warped.png')
    # save_image(im1_warped[0], 'im1_warped.png')
    # save_image(mask_fw[0], 'mask_fw.png')
    # save_image(mask_bw[0], 'mask_bw.png')
    # save_image(diff_im1_y[0], 'diff_im1_y.png')
    # save_image(diff_im1_x[0], 'diff_im1_x.png')
    # save_image(diff_flow_fw_y[0], 'diff_flow_fw_y.png')
    # save_image(diff_flow_fw_y[0], 'diff_flow_fw_x.png')
    # save_image(sec_diff_flow_fw_y[0], 'sec_diff_flow_fw_y.png')
    # save_image(sec_diff_flow_fw_x[0], 'sec_diff_flow_fw_x.png')
    # return

    fb_loss = fb_weight * charbonnier_loss_unflow(flow_diff_fw, mask=mask_fw_2, alpha=fb_exp) + \
              fb_weight * charbonnier_loss_unflow(flow_diff_bw, mask=mask_bw_2, alpha=fb_exp)

    pl_loss = pl_weight * charbonnier_loss_unflow(im2_warped - im1, mask=mask_fw_3, alpha=pl_exp) + \
              pl_weight * charbonnier_loss_unflow(im1_warped - im2, mask=mask_bw_3, alpha=pl_exp)

    sl_loss = sl_weight * charbonnier_loss_unflow(diff_flow_fw_y, mask=mask_fw_2_crop_1_y, alpha=sl_exp) + \
              sl_weight * charbonnier_loss_unflow(diff_flow_fw_x, mask=mask_fw_2_crop_1_x, alpha=sl_exp) + \
              sl_weight * charbonnier_loss_unflow(diff_flow_bw_y, mask=mask_bw_2_crop_1_y, alpha=sl_exp) + \
              sl_weight * charbonnier_loss_unflow(diff_flow_bw_x, mask=mask_bw_2_crop_1_x, alpha=sl_exp)

    sec_sl_loss = sec_sl_weight * charbonnier_loss_unflow(sec_diff_flow_fw_y, mask=mask_fw_2_crop_2_y, alpha=sec_sl_exp) + \
                  sec_sl_weight * charbonnier_loss_unflow(sec_diff_flow_fw_x, mask=mask_fw_2_crop_2_x, alpha=sec_sl_exp) + \
                  sec_sl_weight * charbonnier_loss_unflow(sec_diff_flow_bw_y, mask=mask_bw_2_crop_2_y, alpha=sec_sl_exp) + \
                  sec_sl_weight * charbonnier_loss_unflow(sec_diff_flow_bw_x, mask=mask_bw_2_crop_2_x, alpha=sec_sl_exp)

    wsl_loss = wsl_weight * torch.mean(abs(diff_flow_fw_y) * exp_im1_y) + \
               wsl_weight * torch.mean(abs(diff_flow_fw_x) * exp_im1_x) + \
               wsl_weight * torch.mean(abs(diff_flow_bw_y) * exp_im2_y) + \
               wsl_weight * torch.mean(abs(diff_flow_bw_x) * exp_im2_x)

    loss_dict = {'fb_loss': fb_loss * weight,
                 'pl_loss': pl_loss * weight,
                 'sl_loss': sl_loss * weight,
                 'sec_sl_loss': sec_sl_loss * weight,
                 'wsl_loss': wsl_loss * weight}
    return loss_dict


def forward_backward_loss(im1, im2, flow_fw, flow_bw, config):
    fb_weight = config['fb_weight']
    fb_exp = config['fb_exp']

    flow_fw = F.interpolate(input=flow_fw, scale_factor=im1.shape[2]/flow_fw.shape[2], mode='bilinear')
    flow_bw = F.interpolate(input=flow_bw, scale_factor=im2.shape[2]/flow_bw.shape[2], mode='bilinear')

    im2_warped, mask_fw = image_warp(im2, flow_fw, with_mask=True)
    im1_warped, mask_bw = image_warp(im1, flow_bw, with_mask=True)

    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.01 * mag_sq_bw + 0.5

    fb_occ_fw = (length_sq(flow_diff_fw) > occ_thresh_fw).float()
    fb_occ_bw = (length_sq(flow_diff_bw) > occ_thresh_bw).float()
    mask_fw *= (1 - fb_occ_fw)
    mask_bw *= (1 - fb_occ_bw)

    return fb_weight * charbonnier_loss_unflow(flow_diff_fw, mask=mask_fw, alpha=fb_exp) + \
           fb_weight * charbonnier_loss_unflow(flow_diff_bw, mask=mask_bw, alpha=fb_exp)

def photometric_loss(im1, im2, flow, config):
    """ calculating photometric loss by warping im2 with flow (or im1 with flow for negative case)
    """
    pl_weight = config['pl_weight']
    pl_exp = config['pl_exp']

    warped_image = image_warp(im2, flow)

    # apply charbonnier loss
    return pl_weight * charbonnier_loss(warped_image - im1, pl_exp)


def smoothness_loss(flow, config):
    sl_weight = config['sl_weight']
    sl_exp = config['sl_exp']

    diff_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    diff_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    return sl_weight * charbonnier_loss_unflow(diff_y) + \
           sl_weight * charbonnier_loss_unflow(diff_x)


def weighted_smoothness_loss(im1, im2, flow, config):
    # calculates |grad U_x| * exp(-|grad I_x|) +
    #            |grad U_y| * exp(-|grad I_y|) +
    #            |grad V_x| * exp(-|grad I_x|) +
    #            |grad V_y| * exp(-|grad I_y|)

    sl_weight = config['sl_weight']
    image = im1

    # todo: no idea if downsampling or upsampling is better...
    if image.shape[2] != flow.shape[2]:
        image = F.interpolate(input=image, scale_factor=flow.shape[2]/im1.shape[2], mode='bilinear').to(device)

    diff_flow_y = abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    diff_flow_x = abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

    diff_img_y = abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    diff_img_x = abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    exp_y = torch.exp(-torch.mean(diff_img_y, dim=1, keepdim=True)).expand(-1,2,-1,-1)
    exp_x = torch.exp(-torch.mean(diff_img_x, dim=1, keepdim=True)).expand(-1,2,-1,-1)

    return sl_weight*torch.mean((diff_flow_y * exp_y)) + \
           sl_weight*torch.mean((diff_flow_x * exp_x))


def charbonnier_loss_unflow(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.
    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """

    error = torch.pow(torch.pow(x * beta,2) + epsilon**2, alpha)

    if mask is not None:
        error = torch.mul(mask, error)

    if truncate is not None:
        error = torch.min(error, truncate)

    return torch.mean(error)
