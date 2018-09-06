"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer, MUNIT_Trainer
from data_io.choose_dataset import choose_dataset
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--ne_weight', type=float, default=-1.0)
parser.add_argument('--origin', action='store_true')
parser.add_argument('--cyc_rec_weight', type=float, default=-1.0)
parser.add_argument('--rec_c_weight', type=float, default=-1.0)
parser.add_argument('--zero_rec', action='store_true')
parser.add_argument('--no_rec_s', action='store_true')
parser.add_argument('--no_style_enc', action='store_true')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

config['gen']['style_dim'] = config['gen']['z_num']
config['gen']['no_style_enc'] = opts.no_style_enc
config['origin'] = opts.origin
config['zero_z'] = opts.zero_rec
config['no_rec_s'] = opts.no_rec_s

if opts.cyc_rec_weight != -1:
    config['recon_x_cyc_w'] = opts.cyc_rec_weight
if opts.ne_weight != -1:
    config['loss_eg_weight'] = opts.ne_weight
if opts.rec_c_weight != -1:
    config['recon_c_w'] = opts.rec_c_weight



# Setup model and data loader
if opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
elif opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)

trainer.cuda()

Dataset = choose_dataset(config['dataset_name'])
dataset = Dataset(config['data_root'], config, split='train')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config['batch_size'],
    drop_last=True, shuffle=True, num_workers=int(config['num_workers']))


# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0] + '_vgg_%s_%s' \
        % (opts.trainer, opts.suffix)
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path, model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'config.json'), 'w') as f:
    json.dump(config, f)

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b, image_path) in enumerate(dataloader):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0 or \
                (opts.debug and iterations % 20 == 0):
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0 or \
                (opts.debug and iterations % 20 == 0):
            train_display_images_a = torch.cat([dataset.sample_a() \
                for _ in range(int(display_size/3))], dim=0).cuda()
            train_display_images_b = torch.cat([dataset.sample_b() \
                for _ in range(int(display_size/3))], dim=0).cuda()
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0 or \
        (opts.debug and iterations % 20 == 0):
            train_display_images_a = torch.cat([dataset.sample_a() \
                for _ in range(int(display_size/3))], dim=0).cuda()
            train_display_images_b = torch.cat([dataset.sample_b() \
                for _ in range(int(display_size/3))], dim=0).cuda()
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0 or \
        (opts.debug and iterations % 20 == 0):
            trainer.save(checkpoint_directory, iterations)

        iterations += 1

        if iterations >= max_iter or \
        (opts.debug and iterations % 20 == 0):
            sys.exit('Finish training')

