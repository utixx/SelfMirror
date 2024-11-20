import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import torch
from torch import nn

from model import UNet3D
from new_funcs_for_train import train, patch_volume_for_training

#
if __name__ == '__main__':
    # %% Setup some parameters for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_store_dir', default=r'F:\1Project_Denoising\0_Datasets\NAOMI_Norph',
                        type=str, required=False)
    parser.add_argument('--dataset_name', default='NAOMI_Morph_P4', type=str, required=False)
    parser.add_argument('--pth_store_dir', default='./pth', type=str, required=False)

    # dataset preparation for training
    parser.add_argument('--patch_z', default=64, type=int, required=False, help='patch size in z')
    parser.add_argument('--patch_y', default=64, type=int, required=False, help='patch size in y')
    parser.add_argument('--patch_x', default=64, type=int, required=False, help='patch size in x')
    parser.add_argument('--overlap_factor', default=0.5, type=float, required=False)
    parser.add_argument('--select_z_size_for_train', default=1000, type=int, required=False)
    parser.add_argument('--set_datasets_size_for_train', default=7000, type=int, required=False)

    parser.add_argument('--scale_factor', default=1, type=float, required=False, help='vm scaling factor')

    # %% Setup some parameters for training
    parser.add_argument('--lr', default=0.0001, type=float, required=False, help='learning rate')
    parser.add_argument('--ngpu', default=1, type=int, required=False, help='number of gpu')
    parser.add_argument('--epoch_num', default=20, type=int, required=False, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch size for training')  # 8
    parser.add_argument('--num_workers', default=0, type=int, required=False, help='num_workers for dataloader')  # 4

    args = parser.parse_args()

    args.hr_vm_list = None

    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    args.gap_z = int(args.patch_z * (1 - args.overlap_factor))  # patch gap in z

    # %% Path makedirs
    args.dataset_path = os.path.join(args.datasets_store_dir, args.dataset_name)

    pth_name = args.dataset_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    args.pth_path = os.path.join(args.pth_store_dir, pth_name)
    if not os.path.exists(args.pth_path):
        os.makedirs(args.pth_path)

    # %% volume patching
    print('\033[1;31mPreparing volume images-----> \033[0m')
    (args.patches_name_list,
     args.patches_gcoordinate_list,
     args.volume_index,
     args.noise_vm_all) = patch_volume_for_training(args)

    print('--Patch_nums for training: ', len(args.patches_name_list))

    # %% Model preparation
    print('\033[1;31mLoading network-----> \033[0m')
    # model = UNet3D(in_channels=1, f_maps=[16, 32, 64, 128])
    model = UNet3D(in_channels=1,
                   out_channels=1,
                   f_maps=16,
                   stage=4,
                   final_sigmoid=True)

    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(args.ngpu))
        print('\033[1;31mLoading {} GPU(s) -----> \033[0m'.format(torch.cuda.device_count()))

    # %% ready to train
    train(args, model)
