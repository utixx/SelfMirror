import os
import sys
sys.path.append(os.getcwd())

import argparse
import datetime

import torch
from torch import nn

from SAP import SAP_UNet
from new_funcs_for_test import test


# from funcs_for_train import train, patch_prepare_for_training


#
def main():
    # %% Setup some parameters for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_store_dir', default=r'/home/ut108/aUT/1_datasets/0_NAOMI_Morph', 
                        type=str, required=False)
    parser.add_argument('--dataset_name', type=str, required=False, default='NAOMI_Morph_P4')
    parser.add_argument('--denoise_models', default='NAOMI_Morph_P4_202410241104')
    parser.add_argument('--pth_store_dir', default='./pth', type=str, required=False)
    parser.add_argument('--output_store_dir', default='./results', type=str, required=False)

    # dataset preparation for testing
    parser.add_argument('--patch_z', default=64, type=int, required=False, help='patch size in z')
    parser.add_argument('--patch_y', default=64, type=int, required=False, help='patch size in y')
    parser.add_argument('--patch_x', default=64, type=int, required=False, help='patch size in x')
    # large overlap needed to eliminate the stitching zigzag
    parser.add_argument('--overlap_factor', default=0.6, type=float, required=False,
                        help='overlap factor of two patches')

    parser.add_argument('--select_z_for_test', default=10000, type=int, required=False,
                        help='maximum size in z selected for testing')
    parser.add_argument('--scale_factor', default=1, type=float, required=False, help='vm scaling factor')

    # %% Setup some parameters for training
    parser.add_argument('--ngpu', default=1, type=int, required=False)
    parser.add_argument('--batch_size', default=1, type=int, required=False)
    parser.add_argument('--num_workers', default=0, type=int, required=False)

    args = parser.parse_args()

    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    args.gap_z = int(args.patch_z * (1 - args.overlap_factor))  # patch gap in z

    # %% Path makedirs
    args.dataset_path = os.path.join(args.datasets_store_dir, args.dataset_name)

    if not os.path.exists(args.output_store_dir):
        os.mkdir(args.output_store_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_data_name = 'Dataset-' + args.dataset_name + '_' + current_time + '_Model-' + args.denoise_models
    args.output_path = os.path.join(args.output_store_dir, output_data_name)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # %% read img:
    noise_vm_list = list(os.walk(args.dataset_path, topdown=False))[-1][-1]
    noise_vm_list.sort()
    print('\033[1;31mPreparing volume images-----> \033[0m')
    print('--Total volume number -----> ', len(noise_vm_list))
    for vm_i in noise_vm_list:
        print('        ----->', vm_i)

    args.noise_vm_list = noise_vm_list

    # %% read model list:
    model_path = os.path.join(args.pth_store_dir, args.denoise_models)
    model_list = list(os.walk(model_path, topdown=False))[-1][-1]
    model_list.sort()
    print('\033[1;31mPreparing model weights-----> \033[0m')

    for i in range(len(model_list)):
        aaa = model_list[i]
        if '.csv' in aaa:
            del model_list[i]
    model_list.sort()
    model_list[:-1] = []
    args.model_list = model_list

    # %% Model preparation
    print('\033[1;31mLoading network-----> \033[0m')
    # model = UNet3D(in_channels=1, f_maps=[16, 32, 64, 128])
    model = SAP_UNet(in_channels=1, feat_list=[32, 64, 128], center_feat=64)

    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(args.ngpu))
        print('\033[1;31mLoading {} GPU(s) -----> \033[0m'.format(torch.cuda.device_count()))

    test(args, model)


if __name__ == '__main__':
    main()
