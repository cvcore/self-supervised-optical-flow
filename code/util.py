import os
import numpy as np
import shutil
import torch
from imageio import imwrite

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value=None):
    if type(flow_map) is not np.ndarray:
        flow_map_np = flow_map.detach().cpu().numpy()
    else:
        flow_map_np = flow_map

    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def save_image(image, file_name='default.png'):
    # convert from tensor to numpy array
    if torch.is_tensor(image):
        image = image.squeeze().cpu().detach().numpy()

    # swap axes
    if image.shape[0] <= 3:
        image = np.moveaxis(image, 0, -1)

    # rough channelwise normalization
    for i in range(3):
        min_val = np.amin(image[:,:,i])
        image[:,:,i] = image[:,:,i] - min_val
        max_val = np.amax(image[:,:,i])
        image[:,:,i] = image[:,:,i] / max_val

    imwrite(file_name, image)
