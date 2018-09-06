"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04, find_latest_model_file, \
        tensor_to_image
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from data_io.choose_dataset import choose_dataset
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from tqdm import trange, tqdm
import numpy as np
from skimage.io import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--output_folder', type=str, help="output target folder")
parser.add_argument('--checkpoint', type=str, 
        default='', help="checkpoint of autoencoders")
# parser.add_argument('--a2b', action='store_true', help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

############### Self-added flag ##################
parser.add_argument('--no_style_enc', action='store_true')

opts = parser.parse_args()

############### Pre-defined Variable
__SAMPLE_NUM = 10
__EVAL_NUM = 30

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
# if not os.path.exists(opts.output_folder):
    # os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
config['gen']['no_style_enc'] = opts.no_style_enc

# Setup model and data loader
config['vgg_w'] = 0

if opts.trainer == 'MUNIT':
    config['gen']['style_dim'] = config['gen']['z_num']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

############## configure checkpoint from output_folder
checkpoint_path = find_latest_model_file(os.path.join(opts.output_folder, 
    'checkpoints'), opts.checkpoint, keyword='gen')

try:
    state_dict = torch.load(checkpoint_path)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(checkpoint_path))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

##### Dataset building

Dataset = choose_dataset(config['dataset_name'])
dataset = Dataset(config['data_root'], config, split='val')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1,
    drop_last=True, shuffle=False, num_workers=int(config['num_workers']))

im_h = config['crop_image_height']
im_w = config['crop_image_height'] * config['im_ratio']

a2b_sample_array = np.zeros((__EVAL_NUM * im_h, (__SAMPLE_NUM+1) * im_w, 3), 
        dtype=np.uint8)
b2a_sample_array = np.zeros((__EVAL_NUM * im_h, (__SAMPLE_NUM+1) * im_w, 3), 
        dtype=np.uint8)

pbar = tqdm(total=__EVAL_NUM)

with torch.no_grad():
     
    for i, (image_a, image_b, _) in enumerate(dataloader):
        image_a = image_a.cuda()
        image_b = image_b.cuda()

        pbar.update(1)
        if i == __EVAL_NUM: break;

        a2b_sample_array[i*im_h:(i+1)*im_h, 0:im_w, :] = \
                tensor_to_image(image_a)
     
        b2a_sample_array[i*im_h:(i+1)*im_h, 0:im_w, :] = \
                tensor_to_image(image_b)

        for j in range(__SAMPLE_NUM):
            output_images = trainer.sample(image_a, image_b, training=False)
            im_array_a2b = tensor_to_image(output_images[3])
            im_array_b2a = tensor_to_image(output_images[-1])

            a2b_sample_array[i*im_h:(i+1)*im_h, (j+1)*im_w:(j+2)*im_w, :] = \
                    im_array_a2b
         
            b2a_sample_array[i*im_h:(i+1)*im_h, (j+1)*im_w:(j+2)*im_w, :] = \
                    im_array_b2a

    
    imsave(os.path.join(opts.output_folder, "a2b_samples.jpg"), a2b_sample_array)
    imsave(os.path.join(opts.output_folder, "b2a_samples.jpg"), b2a_sample_array)

pbar.close()
