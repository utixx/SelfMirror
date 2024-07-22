import argparse
import datetime
import os

import torch
from torch import nn

from UNet3D import UNet3D
from funcs_for_test import read_imglist, test


# from funcs_for_train import train, patch_prepare_for_training


#
def main():
    # %% Setup some parameters for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', default=r'../1_datasets/1_2_Neuron_Cell1_Merge',
                        type=str, required=False, help='dataset path')

    parser.add_argument('--pth_dir', default='./pth', type=str, required=False, help='model checkpoints/ weights path')
    parser.add_argument('--denoise_model', default='1_2_Neuron_Cell1_Merge_202405182320')

    parser.add_argument('--output_dir', default='./results', type=str, required=False, help='prediction directory')
    parser.add_argument('--save_test_images_per_epoch', default=True, type=bool, required=False, help='')

    # dataset preparation for training
    parser.add_argument('--overlap_factor', default=0.75, type=float, required=False,     # large overlap needed to eliminate the stitching zigzag
                        help='overlap factor of two patches')
    parser.add_argument('--patch_z', default=64, type=int, required=False, help='patch size in z')
    parser.add_argument('--patch_y', default=64, type=int, required=False, help='patch size in y')
    parser.add_argument('--patch_x', default=64, type=int, required=False, help='patch size in x')

    # parser.add_argument('--train_datasets_size', default=6000, type=int, required=False, help='maximum patches for
    # training/ training dataset size')
    parser.add_argument('--select_img_num', default=1000, type=int, required=False, help='maximum size in z/t selected')
    parser.add_argument('--test_datasize', default=1000, type=int, required=False,
                        help='maximum size in z selected for validation')
    parser.add_argument('--scale_factor', default=1, type=float, required=False, help='image scaling factor')

    # %% Setup some parameters for training
    # parser.add_argument('--lr', default=0.00005, type=float, required=False, help='learning rate')
    parser.add_argument('--ngpu', default=1, type=int, required=False, help='number of gpu')

    # parser.add_argument('--epoch_num', default=40, type=int, required=False, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch size for training')
    parser.add_argument('--num_workers', default=0, type=int, required=False, help='number of workers for dataloader')

    args = parser.parse_args()

    # args.hr_image = None

    args.gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    args.gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    args.gap_z = int(args.patch_z * (1 - args.overlap_factor))  # patch gap in z

    # %% read_modellist(args):
    """
    Make data folder to store testing results \n
    Important Fields:
        args.datasets_name: the sub folder of the dataset \n
        args.pth_path: the folder for pth file storage \n
    """
    if args.datasets_path[-1] != '/':
        args.datasets_name = args.datasets_path.split("/")[-1]
    else:
        args.datasets_name = args.datasets_path.split("/")[-2]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    args.output_path = args.output_dir + '//' + 'DataFoderIs' + args.datasets_name + '_' + current_time + '_ModelFolderIs_' + args.denoise_model
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # %% read_modellist(args):
    args.img_list = read_imglist(args)  # reading dataset for denoising

    # %% read_modellist(args):
    model_path = args.pth_dir + '//' + args.denoise_model
    model_list = list(os.walk(model_path, topdown=False))[-1][-1]
    model_list.sort()
    # print(model_list)
    model_list = model_list[-7::2]
    # print(model_list)
    # raise "stop"

    # calculate the number of model file
    count_pth = 0
    for i in range(len(model_list)):
        aaa = model_list[i]
        if '.pth' in aaa:
            count_pth = count_pth + 1
    args.model_list = model_list
    args.model_list_length = count_pth

    # %% Model preparation
    model = UNet3D(in_channels=1, f_maps=[16, 32, 64, 128], final_sigmoid_for_test=False)

    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=range(args.ngpu))
        print('\033[1;36m--Using {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))

    test(args, model)


if __name__ == '__main__':
    main()
