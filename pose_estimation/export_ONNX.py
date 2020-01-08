# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Roman Tsyganok (iskullbreakeri@gmail.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from utils.utils import create_logger

import yaml
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Exporting network')
  
    args, rest = parser.parse_known_args()

	# Model Data
	
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)


    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):

    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file



def main():
    args = parse_args()
    # update config
    update_config(args.cfg)
    reset_config(config, args)
	# output dir path
    onnx_path = './models/onnx/'
	
    logger, final_output_dir, tb_log_dir = create_logger(
    config, args.cfg, 'convert')
    if not os.path.isdir(onnx_path):
        logger.info('Creating ' + onnx_path + 'folder...')
        os.makedirs(onnx_path);
    height = 0
    width = 0
	
    with open(args.cfg) as file:
        documents = yaml.load(file, Loader=yaml.FullLoader)
        height = documents['MODEL']['IMAGE_SIZE'][0]
        width = documents['MODEL']['IMAGE_SIZE'][1]
        		


    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )


    logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
    filename = config.TEST.MODEL_FILE
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    logger.info('=> Converting...')
    data = torch.zeros((1, 3, height, width)).cuda()
    model.cuda()
    model.float()
    head, filename = os.path.split(filename)
    if 'pth.tar' in filename:
        filename = filename[:-8]
			
    torch.onnx.export(model, data, onnx_path + filename +'.onnx')
    logger.info('=> Model saved as: ' + onnx_path + filename +'.onnx')
    logger.info('=> Done.')

		



if __name__ == '__main__':
    main()
