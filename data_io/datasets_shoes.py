from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

class Dataset(data.Dataset): 
       
    def __init__(self, data_root, params, split='train'):
        self.config = params


        self.data = []
        self.data_dir = data_root
        self.split = split
        self.split_dir = os.path.join(data_root, 'edges2shoes', split)

        self.filenames = os.listdir(self.split_dir)
        self.dataset_size = len(self.filenames)

        self.simple_trans = transforms.Compose(
                [

                    transforms.Scale([params['crop_image_height'], 
                        params['crop_image_width']*params['im_ratio']]
                         , Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )


    def __getitem__(self, index):
        file_name = self.filenames[index]

        random_index = random.randint(0, self.dataset_size - 1)

        # captions = self.captions[key]
        whole_image_path = os.path.join(self.split_dir, file_name)
        whole_image = Image.open(whole_image_path)
        whole_array = np.asarray(whole_image)
        label_array = whole_array[:, :256, :]

        whole_image_path = os.path.join(self.split_dir, self.filenames[random_index])
        whole_image = Image.open(whole_image_path)
        whole_array = np.asarray(whole_image)
        image_array = whole_array[:, 256:, :]


        X_img = Image.fromarray(label_array)
        Y_img = Image.fromarray(image_array)
        ############################################################
        # transoform RGB image, and to Tensor
        # note should use the same transoform param dictionary
        ############################################################
        trans_params = get_param_dict(self.config)

        img_transform = get_transforms(trans_params, method=Image.BILINEAR, 
                normalize=True)

        img_tensor = img_transform(X_img)
        target_img_tensor = img_transform(Y_img)


        return target_img_tensor, img_tensor, file_name


    def sample_a(self):
        sample_tensor = self.simple_trans(self.__get_image(stamp='a'))
        return torch.stack([sample_tensor,] * 3)

    def sample_b(self):
        sample_tensor = self.simple_trans(self.__get_image(stamp='b'))
        return torch.stack([sample_tensor,] * 3)

    def __get_image(self, stamp=None):
        index = random.randint(0, self.dataset_size - 1)
        file_name = self.filenames[index]
        whole_image_path = os.path.join(self.split_dir, file_name)
        whole_image = Image.open(whole_image_path)
        whole_array = np.asarray(whole_image)

        if stamp == "a":
            img = Image.fromarray(whole_array[:, 256:, :])
        elif stamp == "b":
            img = Image.fromarray(whole_array[:, :256, :])


        return img


    def __len__(self):
        return len(self.filenames)

"""assitance function for transforming images"""
def construct_one_hot(label_map_tensor):
    lm_size = label_map_tensor.size()
    oneHot_size = (cfg.CLASS_NUM, lm_size[1], lm_size[2])
    mask_tensor = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    mask_tensor = mask_tensor.scatter_(0, label_map_tensor.long(), 1.0)
    # the 0 represents unlabled. 
    return mask_tensor


def get_param_dict(config):
    params = {}
    params['flip'] = np.random.rand(1)[0] > 0.5


    diff_length, diff_width = config['new_size'] - config['crop_image_height'], \
            (config['new_size'] - config['crop_image_width']) * config['im_ratio']


    pos_x = np.random.randint(0, diff_width)
    pos_y = np.random.randint(0, diff_length)


    params['crop_pos'] = (pos_x, pos_y)
    params['load_size'] = config['new_size']
    params['img_size'] = config['crop_image_height']
    params['im_ratio'] = config['im_ratio']

    return params

"""if we train higher resolution image based on lower resolution pre
    -trained results, we should scale the dict"""
def scale_param_dict(params):
    scaled_params = {}
    pos = params['crop_pos']
    scaled_params['crop_pos'] = (pos[0] * 2, pos[1] * 2)
    scaled_params['load_size'] = params['load_size'] * 2
    scaled_params['img_size'] = params['img_size'] * 2
    scaled_params['flip'] = params['flip']

    return scaled_params

def get_transforms(params, method=Image.BILINEAR, normalize=False):


    trans_list = [
                    transforms.Scale([params['load_size'], 
                        params['load_size']*params['im_ratio']]
                         , method),
                    transforms.Lambda(lambda img: __flip(img, params['flip'])),
                    transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 
                        params['img_size'])),
                    transforms.ToTensor(),
                ]


    if normalize:
        trans_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


    trans = transforms.Compose(trans_list)


    return trans


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw , th = size, size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))


    return img


def classwise_noise(mask_var_cpu, n_num):
    # n_num, noise dim
    b_mask_numpy = mask_var_cpu.clone().numpy()
    bool_mask = b_mask_numpy.astype(np.bool)
    shape = b_mask_numpy.shape
    class_num = shape[0]


    noise_block = np.zeros((n_num, shape[1], shape[2]))


    for j in range(class_num):
        mask_index = bool_mask[j]
        if np.any(mask_index):
            noise_block[:, mask_index] = \
                    np.random.randn(n_num, 1)


    noise_block_tensor = torch.from_numpy(noise_block).clone()


    return noise_block_tensor.float()


def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    return edge.float()
