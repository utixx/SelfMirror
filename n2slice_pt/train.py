import argparse
import datetime
import os

import torch
from torch import nn

from UNet3D import UNet3D
from funcs_for_train import train, patch_prepare_for_training


#
def main():
    # %% Setup some parameters for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', default=r'../1_datasets/2.2_Fig_2_Neuron_Population',
                        type=str, required=False, help='dataset path')

    parser.add_argument('--pth_dir', default='./pth',
                        type=str, required=False, help='model checkpoints/ weights path')

    parser.add_argument('--output_dir', default='./results',
                        type=str, required=False, help='prediction directory')

    # dataset preparation for training
    parser.add_argument('--overlap_factor', default=0.5, type=float, required=False,
                        help='overlap factor of two patches')
    parser.add_argument('--patch_z', default=64, type=int, required=False, help='patch size in z')
    parser.add_argument('--patch_y', default=64, type=int, required=False, help='patch size in y')
    parser.add_argument('--patch_x', default=64, type=int, required=False, help='patch size in x')

    parser.add_argument('--train_datasets_size', default=7000, type=int, required=False,
                        help='maximum patches for training/ training dataset size')
    parser.add_argument('--select_img_num', default=1000, type=int, required=False, help='maximum size in z/t selected')
    parser.add_argument('--test_datasize', default=1000, type=int, required=False,
                        help='maximum size in z selected for validation')
    parser.add_argument('--scale_factor', default=1, type=float, required=False, help='image scaling factor')

    # %% Setup some parameters for training
    parser.add_argument('--lr', default=0.00005, type=float, required=False, help='learning rate')
    parser.add_argument('--ngpu', default=1, type=int, required=False, help='number of gpu')

    parser.add_argument('--epoch_num', default=20, type=int, required=False, help='number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='batch size for training')
    parser.add_argument('--num_workers', default=4, type=int, required=False, help='number of workers for dataloader')

    args = parser.parse_args()

    args.hr_image = None

    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    args.gap_z = int(args.patch_z * (1 - args.overlap_factor))  # patch gap in z

    # %% Model preparation
    model = UNet3D(in_channels=1, f_maps=[16, 32, 64, 128])

    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(args.ngpu))
        print('\033[1;36m--Using {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))

    # %% File path makedir
    """
        Make data folder to store training results
        Important Fields:
            args.datasets_name: the sub folder of the dataset.  \n
            args.pth_path: the folder for pth file storage
    """
    if args.datasets_path[-1] != '/':
        args.datasets_name = args.datasets_path.split("/")[-1]
    else:
        args.datasets_name = args.datasets_path.split("/")[-2]

    pth_name = args.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    args.pth_path = args.pth_dir + '/' + pth_name.split(os.sep)[-1]

    if not os.path.exists(args.pth_path):
        os.makedirs(args.pth_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # %% stack patching
    args.name_list, args.coordinate_list, args.stack_index, args.noise_im_all = patch_prepare_for_training(args)
    print('patch_nums for training: ', len(args.name_list))

    # %% ready to train
    train(args, model)


if __name__ == '__main__':
    main()
