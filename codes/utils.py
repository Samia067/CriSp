import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from collections.abc import Iterable
import os
import matplotlib.pyplot as plt
from torchvision.utils import *
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_compressed_mask(mask):
    return (F.interpolate(mask.float(), (6, 12), mode='bilinear') > 0)

def get_features(inp, net, spatial_feat=False, mask=None):
    return net(inp, spatial_feat=spatial_feat, mask=mask)

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


def get_image_from_tensor(tensor, index=None, gray=False):
    fill_value = 1 if tensor[0,0,0,0] == 0 else 0
    if index is not None:
        t = tensor[index, ...].cpu().detach() # *255
    else:
        h = tensor.shape[2]
        w = tensor.shape[3]
        count = tensor.shape[0]
        t = torch.zeros((tensor.shape[1], h, count*w + count - 1))
        # print(t.shape)
        for i in range(tensor.shape[0]):
            t[:, :, i * (w+1):(i + 1) * (w+1) - 1] = tensor[i, ...] # t[:,:, i*w:(i+1)*w] = tensor[i, ...]
            t[:, :, (i + 1) * (w+1) - 1: (i + 1) * (w+1)] = fill_value
        t = t.cpu().detach()
    if t.shape[0] == 1:
        if gray:
            t = t.repeat(3, 1, 1)
            t = t.transpose(0, 2).transpose(1, 0)
        else:
            t = t[0, ...]
    else:
        t = t.transpose(0, 2).transpose(1, 0)
    t = t.numpy()
    return t

def save_tensor_grid(data, save_path, BGR_to_RGB=True, fig_shape='square', figsize=None):
    show_tensor_grid(data, fig_shape=fig_shape, figsize=figsize)
    plt.savefig(save_path)
    plt.close()

def show_tensor_grid(data, BGR_to_RGB=True, fig_shape='square', figsize=None):
    if fig_shape =='square':
        fig_shape = (int(np.ceil(np.sqrt(len(data)))), int(np.ceil(np.sqrt(len(data)))))
    elif fig_shape == 'line':
        fig_shape = (len(data), 1)
    else:
        assert fig_shape[0]*fig_shape[1] >= len(data)
    if figsize is None:
        figsize = (10, 10)

    fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=figsize)
    for index, title in enumerate(data.keys()):
        ax_coord = np.unravel_index(index, fig_shape)
        image = get_image_from_tensor(data[title])
        if BGR_to_RGB and len(image.shape) == 3:
            image = image[..., [2, 1, 0]]
        if 1 in fig_shape:
            axs[ax_coord[0]].imshow(image)
            axs[ax_coord[0]].set_title(title)
        else:
            axs[ax_coord[0], ax_coord[1]].imshow(image)
            axs[ax_coord[0], ax_coord[1]].set_title(title)
    fig.tight_layout()
    return fig


"""Return a placement tensor for an invalid tensor"""
def get_invalid_tensor(tensor=True):
    if tensor:
        return torch.Tensor([-1]).int()
    else:
        return -1

"""Check if a tensor is invalid"""
def invalid_tensor(tensor):
    return (tensor is None) or torch.all(tensor == -1)

"""Check if a tensor is valid"""
def valid_tensor(tensor):
    return not invalid_tensor(tensor)


def prepare_input(database_inp, mask, opt, image1, depth1, print1):
    inp_types = opt.input if (database_inp or opt.input2 is None) else opt.input2
    inp_index_dim1 = 1 if database_inp and (inp_types[0] != 'print') else 0

    inp_map = {'rgb': image1 if valid_tensor(image1) else None,
               'depth': depth1 if valid_tensor(depth1) else None,
               'print': print1 if valid_tensor(print1) else None}

    valid_key = [key for key in inp_map.keys() if inp_map[key] is not None][0]
    valid_shape = inp_map[valid_key].shape
    inp = torch.zeros((valid_shape[0], opt.in_channel, valid_shape[2], valid_shape[3])).to(opt.device)

    for input_type in inp_types:
        cur_channel_count = inp_map[input_type].shape[1]
        if mask is not None:
            item = inp_map[input_type]
            fill = torch.median(torch.tensor([item[0,0,0,0], item[0,0,0,-1], item[0,0,-1,-1], item[0,0,-1,0]]))
            item[~mask.expand_as(item)] = fill

        inp[:, inp_index_dim1:inp_index_dim1+cur_channel_count, ...] = inp_map[input_type]
        inp_index_dim1 += cur_channel_count

    return inp

def get_shoeid_str(name):
    return name.split('_', 3)[2]

def load_model(save_file, model, optimizer=None):
    print('==> Loading ', save_file)
    file = torch.load(save_file)
    model.load_state_dict(file['model'])
    if optimizer:
        optimizer.load_state_dict(file['optimizer'])
    return model, optimizer, file['opt'], file['epoch']
