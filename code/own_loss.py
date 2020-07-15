import torch
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def charbonnier_loss(input, alpha):
    eps = 1e-9 # from reference implementation
    sq = torch.pow(input,2) + eps*eps
    return torch.mean(torch.pow(sq, alpha))


def photometric_loss(im1, im2, flow, config):
    """ calculating photometric loss by warping im2 with flow (or im1 with flow for negative case)
    """
    pl_weight = config['pl_weight']
    forward_flow = config['forward_flow']
    pl_exp = config['pl_exp']

    # upscaling in case the height does not match. Assumes image ratio is correct
    if im1.shape[2] != flow.shape[2]:
        flow = F.interpolate(input=flow, scale_factor=im1.shape[2]/flow.shape[2], mode='bilinear')

    # adapted from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    if forward_flow:
        image = im1
        image_target = im2
    else:
        image = im2
        image_target = im1

    B, C, H, W = image.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)

    if forward_flow:
        vgrid = grid - flow
    else:
        vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    warped_image = F.grid_sample(image, vgrid)

    # for debug purpose
    # save_image(im1[0], 'im1.png')
    # save_image(im2[0], 'im2.png')
    # save_image((warped_image)[0], 'diff.png')
    # return

    # apply charbonnier loss
    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    if config['use_l1_loss']:
        return pl_weight * F.l1_loss(warped_image, image_target)
    else:
        return pl_weight * charbonnier_loss(warped_image - image_target, pl_exp)



def smoothness_loss(flow, config):
    sl_weight = config['sl_weight']
    sl_exp = config['sl_exp']

    diff_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    diff_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    if config['use_l1_loss']:
        return sl_weight * (F.l1_loss(diff_y, torch.zeros_like(diff_y)) + F.l1_loss(diff_x, torch.zeros_like(diff_x)))
    else:
        return sl_weight * charbonnier_loss(diff_y, sl_exp) + \
               sl_weight * charbonnier_loss(diff_x, sl_exp)

def weighted_smoothness_loss(im1, im2, flow, config):
    # calculates |grad U_x| * exp(-|grad I_x|) +
    #            |grad U_y| * exp(-|grad I_y|) +
    #            |grad V_x| * exp(-|grad I_x|) +
    #            |grad V_y| * exp(-|grad I_y|)

    sl_weight = config['sl_weight']
    forward_flow = config['forward_flow']

    # todo: no idea which image to take...
    if forward_flow:
        image = im2
    else:
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











