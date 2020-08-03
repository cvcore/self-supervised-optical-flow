import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import ssim_module
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def charbonnier_loss(input, alpha):
    eps = 1e-9 # from reference implementation
    sq = torch.pow(input,2) + eps*eps
    return torch.mean(torch.pow(sq, alpha))


def image_warp(image, flow, with_mask=False):
    if image.shape[2] != flow.shape[2]:
        flow = F.interpolate(input=flow, scale_factor=image.shape[2]/flow.shape[2], mode='bilinear')

    B, C, H, W = image.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(device)
    vgrid = grid + flow

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
        return output, mask
    else:
        return output

def photometric_loss(im1, im2, flow, config):
    """ calculating photometric loss by warping im2 with flow (or im1 with flow for negative case)
    """
    pl_weight = config['pl_weight']
    pl_exp = config['pl_exp']

    warped_image = image_warp(im2, flow)

    # for debug purpose
    # save_image(im1[0], 'im1.png')
    # save_image(im2[0], 'im2.png')
    # save_image((warped_image)[0], 'diff.png')
    # return

    # apply charbonnier loss
    if config['use_l1_loss']:
        return pl_weight * F.l1_loss(warped_image, im1)
    else:
        return pl_weight * charbonnier_loss(warped_image - im1, pl_exp)

def length_sq(mat):
    return torch.sum(torch.pow(mat, 2), dim=1, keepdim=True)


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



def smoothness_loss(flow, config):
    sl_weight = config['sl_weight']
    sl_exp = config['sl_exp']

    diff_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    diff_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]

    # magic numbers from https://github.com/ryersonvisionlab/unsupFlownet
    if config['use_l1_loss']:
        return sl_weight * (F.l1_loss(diff_y, torch.zeros_like(diff_y)) + F.l1_loss(diff_x, torch.zeros_like(diff_x)))
    elif config['unflow']:
        return sl_weight * charbonnier_loss_unflow(diff_y) + \
               sl_weight * charbonnier_loss_unflow(diff_x)
    else:
        return sl_weight * charbonnier_loss(diff_y, sl_exp) + \
               sl_weight * charbonnier_loss(diff_x, sl_exp)


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



# unflow losses adapted from the official tensorflow implementation
# https://github.com/simonmeister/UnFlow
def ternary_loss(im1, im2, flow, max_distance=1):

    im_warped = image_warp(im2, flow)
    im_target = im1

    patch_size = 2 * max_distance + 1

    def _ternary_transform(im):
        B, C, H, W = im.size()
        #im = im.permute(0,2,3,1)
        #numpy_arr = np.array([(_.data).cpu().numpy() for _ in im])
        #numpy_arr_gray = np.array([rgb2gray(_).reshape(H, W, 1) for _ in numpy_arr])
        #intensities = torch.tensor(numpy_arr_gray).permute(0, 3, 1, 2).to(device)
        #pil_im = [torchvision.transforms.functional.to_pil_image(_.cpu(), mode=None) for _ in im]
        #intensities = [torchvision.transforms.functional.to_grayscale(pil, num_output_channels=1) for pil in pil_im]
        #intensities = [torchvision.transforms.functional.to_tensor(_).reshape(1,H,W).to(device) for _ in intensities]
        #intensities = torch.stack(intensities,dim=0)
        intensities = im.mean(dim=1,keepdim=True)


        out_channels = patch_size * patch_size
        conv = torch.nn.Conv2d(in_channels=1,out_channels=out_channels, kernel_size=patch_size, stride=1, padding=1).to(device)
        patches = conv(intensities)

        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf,2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow((t1 - t2),2)
        dist_norm = dist / (0.1 + dist)
        dist_sum = torch.sum(dist_norm, 3, keepdim=True)
        return dist_sum

    t1 = _ternary_transform(im_target)
    t2 = _ternary_transform(im_warped)
    dist = _hamming_distance(t1, t2)

    #transform_mask = create_mask(mask, [[max_distance, max_distance],
    #                                    [max_distance, max_distance]])
    return charbonnier_loss_unflow(dist)

def create_mask(tensor, paddings):

    shape = tensor.shape
    inner_width = shape[1] - (paddings[0] + paddings[1])
    inner_height = shape[2] - (paddings[2] + paddings[3])
    inner = torch.ones([inner_width, inner_height])

    mask2d = torch.nn.functional.pad(inner, paddings)
    mask3d = mask2d.unsqueeze(0).repeat([shape[0], 1, 1])
    mask4d = mask3d.unsqueeze(3)
    return mask4d.requires_grad_(False)


def second_order_loss(flow):

    delta_u, delta_v, mask = _second_order_deltas(flow)
    loss_u = charbonnier_loss_unflow(delta_u, mask)
    loss_v = charbonnier_loss_unflow(delta_v, mask)
    return loss_u + loss_v

def _second_order_deltas(flow):
    print(flow.size())
    mask_x = create_mask(flow, (0, 0, 1, 1))
    mask_y = create_mask(flow, (1, 1, 0, 0))
    mask_diag = create_mask(flow, (1, 1, 1, 1))
    mask = torch.cat([mask_x, mask_y, mask_diag, mask_diag],dim=3)

    filter_x = [[0, 0, 0],
                [1, -2, 1],
                [0, 0, 0]]
    filter_y = [[0, 1, 0],
                [0, -2, 0],
                [0, 1, 0]]
    filter_diag1 = [[1, 0, 0],
                    [0, -2, 0],
                    [0, 0, 1]]
    filter_diag2 = [[0, 0, 1],
                    [0, -2, 0],
                    [1, 0, 0]]
    weight_array = np.ones([3, 3, 1, 4])
    weight_array[:, :, 0, 0] = filter_x
    weight_array[:, :, 0, 1] = filter_y
    weight_array[:, :, 0, 2] = filter_diag1
    weight_array[:, :, 0, 3] = filter_diag2

    flow_u, flow_v = torch.split(flow,2,dim=3)
    conv = torch.nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,padding = 1)
    delta_u = conv(flow_u)
    delta_v = conv(flow_v)
    return delta_u, delta_v, mask



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

    batch, height, width, channels = x.shape
    normalization = batch * height * width * channels

    error = torch.pow(torch.pow(x * beta,2) + epsilon**2, alpha)

    if mask is not None:
        error = torch.mul(mask, error)

    if truncate is not None:
        error = torch.min(error, truncate)

    return torch.sum(error) / normalization

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


def ssim(im1,im2,flow):
    im_warped = image_warp(im2, flow)

    ssim_loss = 1.0 - ssim_module.ssim(im1, im_warped, window_size=11, size_average=True)
    return ssim_loss