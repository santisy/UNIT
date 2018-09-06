"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04, find_latest_model_file
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
from tqdm import trange
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--output_folder', type=str, help="output target folder")
parser.add_argument('--checkpoint', type=str, 
        default='', help="checkpoint of autoencoders")
# parser.add_argument('--a2b', action='store_true', help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

############### Self-added flag ##################
parser.add_argument('--no_style_enc', action='store_true')

opts = parser.parse_args()

############### Pre-defined Variable
__TRY_NUM = 10
__SAMPLE_NUM = 10
__EVAL_NUM = 100


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

with torch.no_grad():
     
    # Start testing

    # load the LPIPS model
    from LPIPS.models import dist_model as dm
    dis_model = dm.DistModel()
    dis_model.initialize(model='net-lin',net='alex',use_gpu=True)

    img_collect_a2b = []
    img_collect_b2a = []
    dis_array_a2b = np.zeros(__TRY_NUM)
    dis_array_b2a = np.zeros(__TRY_NUM)

    for j in trange(__TRY_NUM, desc='try_num'):
        dis_accu_a2b = 0
	dis_accu_b2a = 0
	for _ in trange(__EVAL_NUM, desc='eval_num'):

            input_a = (dataset.sample_a().cuda()[0]).unsqueeze(dim=0)
            input_b = (dataset.sample_b().cuda()[0]).unsqueeze(dim=0)

	    for _ in trange(__SAMPLE_NUM, desc='sample_num'):
		while len(img_collect_a2b) != 2:
                    output_images = trainer.sample(input_a, input_b, 
                            training=False)

		    img_collect_a2b.append(output_images[3].data)
		    img_collect_b2a.append(output_images[-1].data)

		d_ab = dis_model.forward(*img_collect_a2b)
		d_ba = dis_model.forward(*img_collect_b2a)
                dis_accu_a2b += d_ab[0]
		dis_accu_b2a += d_ba[0]

		img_collect_a2b = []
		img_collect_b2a = []

	dis_array_a2b[j] = dis_accu_a2b / (__SAMPLE_NUM * __EVAL_NUM)
	dis_array_b2a[j] = dis_accu_b2a / (__SAMPLE_NUM * __EVAL_NUM)

    output_str = '''For total %d pairs of random samples
    from %d condition, trial num %d, 
    the LPIPS score of a2b is %.4f, std %.4f;
    the LPIPS score of b2a is %.4f, std %.4f;
	    .\n''' % (__SAMPLE_NUM * __EVAL_NUM, __EVAL_NUM,
		__TRY_NUM, dis_array_a2b.mean(), dis_array_a2b.std(),
                dis_array_b2a.mean(), dis_array_b2a.std())

    print('''\033[1;31m %s \033[0m''' % output_str)

    with open(os.path.join(opts.output_folder, 'LPIPS_scores.txt'), 'a') as f:
	date_str = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
	f.write('%s, %s' % (date_str, output_str))
