import datetime
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader
# from dataset_generator import testset, multibatch_test_save, singlebatch_test_save
import random
import numpy as np
import os
import tifffile as tiff
from skimage import io
import math
import csv
from tqdm import tqdm

from new_funcs_for_test import Testset, patch_volume_for_test, singlebatch_test_save, multibatch_test_save


#  ###################################################################################
def random_transform(inputs, targets):  # 8 * 2 transforms
    """
    The function for data augmentation.
    """
    trans = random.randrange(8)
    if trans == 0:  # no transformation
        inputs = inputs
        targets = targets
    elif trans == 1:  # left rotate 90
        inputs = np.rot90(inputs, k=1, axes=(1, 2))
        targets = np.rot90(targets, k=1, axes=(1, 2))
    elif trans == 2:  # 180
        inputs = np.rot90(inputs, k=2, axes=(1, 2))
        targets = np.rot90(targets, k=2, axes=(1, 2))
    elif trans == 3:  # 270
        inputs = np.rot90(inputs, k=3, axes=(1, 2))
        targets = np.rot90(targets, k=3, axes=(1, 2))
    elif trans == 4:  # horizontal flip
        inputs = inputs[:, :, ::-1]
        targets = targets[:, :, ::-1]
    elif trans == 5:  # horizontal flip & left rotate 90
        inputs = np.rot90(inputs[:, :, ::-1], k=1, axes=(1, 2))
        targets = np.rot90(targets[:, :, ::-1], k=1, axes=(1, 2))
    elif trans == 6:  # horizontal flip & left rotate 180
        inputs = np.rot90(inputs[:, :, ::-1], k=2, axes=(1, 2))
        targets = np.rot90(targets[:, :, ::-1], k=2, axes=(1, 2))
    elif trans == 7:  # horizontal flip & left rotate 270
        inputs = np.rot90(inputs[:, :, ::-1], k=3, axes=(1, 2))
        targets = np.rot90(targets[:, :, ::-1], k=3, axes=(1, 2))

    trans_1 = random.randrange(2)  # z flip
    if trans_1 == 0:
        inputs = inputs
        targets = targets
    else:
        inputs = inputs[::-1, :, :]
        targets = targets[::-1, :, :]

    trans_2 = random.randrange(2)  # swap inputs targets
    if trans_2 == 0:
        inputs = inputs
        targets = targets
    else:
        inputs, targets = targets, inputs

    return inputs, targets


# ####################################################################################
class Trainset(Dataset):
    def __init__(self, patches_name_list, patches_gcoordinate_list, volume_index, noise_vm_all):
        self.patches_name_list = patches_name_list
        self.patches_gcoordinate_list = patches_gcoordinate_list
        self.volume_index = volume_index  # list
        self.noise_vm_all = noise_vm_all

    def __getitem__(self, index):
        volume_index = self.volume_index[index]
        noise_vm = self.noise_vm_all[volume_index]
        one_patch_gcoordinate = self.patches_gcoordinate_list[self.patches_name_list[index]]

        init_x = one_patch_gcoordinate['init_x']
        end_x = one_patch_gcoordinate['end_x']
        init_y = one_patch_gcoordinate['init_y']
        end_y = one_patch_gcoordinate['end_y']
        init_z = one_patch_gcoordinate['init_z']
        end_z = one_patch_gcoordinate['end_z']

        # #########-------core------------##############

        # p_index = random.randrange(3)
        # if p_index == 0:
        inputs = noise_vm[init_z:end_z:2, init_y:end_y, init_x:end_x]
        targets = noise_vm[init_z + 1:end_z:2, init_y:end_y, init_x:end_x]
        # elif p_index == 1:
        #     inputs = noise_vm[init_z:end_z, init_y:end_y:2, init_x:end_x]
        #     targets = noise_vm[init_z:end_z, init_y + 1:end_y:2, init_x:end_x]
        # elif p_index == 2:
        #     inputs = noise_vm[init_z:end_z, init_y:end_y, init_x:end_x:2]
        #     targets = noise_vm[init_z:end_z, init_y:end_y, init_x + 1:end_x:2]

        #
        inputs, targets = random_transform(inputs, targets)
        inputs = torch.from_numpy(np.expand_dims(inputs, 0).copy())
        targets = torch.from_numpy(np.expand_dims(targets, 0).copy())

        return inputs, targets

    def __len__(self):
        return len(self.patches_name_list)


# ####################################################################################
def patch_volume_for_training(args):
    """
    The original noisy stack is partitioned into 3D patches with the setting
    overlap factor in each dimension.

    Return:
       patches_name_list : the coordinates of 3D patch are indexed by the patch name in args.name_list. \n
       patches_coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack. \n
       volume_index : the index of the noisy stacks. \n
       noise_vm_all : the collection of all noisy stacks. \n
    """

    patch_x, patch_y, patch_z = args.patch_x, args.patch_y, args.patch_z
    gap_x, gap_y = args.gap_x, args.gap_y

    dataset_path, dataset_name = args.dataset_path, args.dataset_name
    select_z_size_for_train = args.select_z_size_for_train
    set_datasets_size_for_train = args.set_datasets_size_for_train

    scale_factor = args.scale_factor

    patches_name_list = []
    patches_gcoordinate_list = {}
    volume_index = []
    noise_vm_all = []
    ind = 0

    print('\033[1;34mImage list for training -----> \033[0m')
    volume_nums = len(list(os.walk(dataset_path, topdown=False))[-1][-1])
    print('--Total stack number -----> ', volume_nums)

    for vm_name in list(os.walk(dataset_path, topdown=False))[-1][-1]:
        print('---Noise image name -----> ', vm_name)
        vm_dir = dataset_path + '//' + vm_name
        noise_vm = tiff.imread(vm_dir)
        print('---Noise image shape -----> ', noise_vm.shape)

        if noise_vm.shape[0] > select_z_size_for_train:  # 超过就截断
            noise_vm = noise_vm[0:select_z_size_for_train, :, :]

        whole_x = noise_vm.shape[2]
        whole_y = noise_vm.shape[1]
        whole_z = noise_vm.shape[0]

        # Calculate real gap_z
        # 计算z或者t轴的步长, 来满足我有多少个patches, args.train_datasets_size参数
        x_nums = math.floor((whole_x - patch_x) / gap_x) + 1
        y_nums = math.floor((whole_y - patch_y) / gap_y) + 1
        z_nums = math.ceil(set_datasets_size_for_train / x_nums / y_nums / volume_nums)  # volm_num是有多少个三维图像
        gap_z = math.floor((whole_z - patch_z) / (z_nums - 1))  # real gap z
        print('---real gap_z--->', gap_z)

        assert gap_z >= 0 and gap_y >= 0 and gap_x >= 0, "train gat size is negative!"

        noise_vm = noise_vm.astype(np.float32) / scale_factor
        noise_vm = noise_vm - noise_vm.mean()  # processing, Minus mean before training

        noise_vm_all.append(noise_vm)

        for z in range(0, int((whole_z - patch_z + gap_z) / gap_z)):
            for y in range(0, int((whole_y - patch_y + gap_y) / gap_y)):
                for x in range(0, int((whole_x - patch_x + gap_x) / gap_x)):
                    one_patch_gcoordinate = {'init_x': 0, 'end_x': 0, 'init_y': 0, 'end_y': 0, 'init_z': 0, 'end_z': 0}

                    init_x = gap_x * x
                    end_x = gap_x * x + patch_x
                    init_y = gap_y * y
                    end_y = gap_y * y + patch_y
                    init_z = gap_z * z
                    end_z = gap_z * z + patch_z

                    one_patch_gcoordinate['init_x'] = init_x
                    one_patch_gcoordinate['end_x'] = end_x
                    one_patch_gcoordinate['init_y'] = init_y
                    one_patch_gcoordinate['end_y'] = end_y
                    one_patch_gcoordinate['init_z'] = init_z
                    one_patch_gcoordinate['end_z'] = end_z

                    patch_name = dataset_name + '_' + vm_name.replace('.tif', '') + '_x' + str(
                        x) + '_y' + str(y) + '_z' + str(z)

                    patches_name_list.append(patch_name)
                    patches_gcoordinate_list[
                        patch_name] = one_patch_gcoordinate  # {patch_name: one_patch_gcoordinate_{}}
                    volume_index.append(ind)  # 00...0011...1122...22
        ind = ind + 1

    return patches_name_list, patches_gcoordinate_list, volume_index, noise_vm_all


def safe_bce_loss(inputs, target, epsilon=1e-7):
    inputs = torch.clamp(inputs, epsilon, 1 - epsilon)
    return F.binary_cross_entropy(inputs, target, reduction='mean')


# ####################################################################################
def train(args, model):
    patches_name_list = args.patches_name_list
    patches_gcoordinate_list = args.patches_gcoordinate_list
    noise_vm_all = args.noise_vm_all
    volume_index = args.volume_index
    lr, batch_size, num_workers, epoch_num = args.lr, args.batch_size, args.num_workers, args.epoch_num
    hr_vm_list = args.hr_vm_list
    pth_path = args.pth_path

    psnr_list = []

    print("\033[1;34mdefine optimizer -----> \033[0m")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    print("\033[1;34mdefine loss func -----> \033[0m")
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()
    # BCE = torch.nn.BCELoss()
    # BCE.cuda()
    BCE = safe_bce_loss  # 使用安全的BCE损失函数
    L2.cuda()
    L1.cuda()

    print("\033[1;34mstart training -----> \033[0m")

    for epoch in range(0, epoch_num):
        model.train()
        # load dataset very epoch ? 好像是因为用了distribute gpu  详见:https://zhuanlan.zhihu.com/p/552476081
        # print(len(train_data))   len(dataloader) = dataset/batch
        train_data = Trainset(patches_name_list, patches_gcoordinate_list, volume_index, noise_vm_all)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        tqdm_train = tqdm(enumerate(trainloader), total=len(trainloader),
                          desc=f"Epoch {epoch+1}/{epoch_num}", leave=True)
        for batch_idx, (inputs, targets) in tqdm_train:
            # print(inputs.shape, targets.shape)
            optimizer.zero_grad()

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs_v = Variable(inputs, requires_grad=False)
            targets_v = Variable(targets, requires_grad=False)

            outputs_in, decoder_outputs_in = model(inputs_v)
            # print(len(decoder_outputs_in))
            outputs_ta, decoder_outputs_ta = model(targets_v)

            l1_loss = L1(outputs_in, targets_v)
            l2_loss = L2(outputs_in, targets_v)
            # loss = 0.5 * l1_loss + 0.5 * l2_loss
            side_loss = 0.0

            for de_out_in, de_out_ta in zip(decoder_outputs_in, decoder_outputs_ta):
                # print(de_out_in.max(), de_out_ta.max())
                # side_loss += BCE(de_out_in, de_out_ta)  # BCE损失期望输入在0到1之间 如果用loss模型增加sigmoid输出
                side_loss += 0.5 * L1(de_out_in, de_out_ta) + 0.5 * L2(de_out_in, de_out_ta)
                # print(side_loss.item())
            # side_loss /= len(decoder_outputs_in)

            loss = l1_loss + l2_loss + side_loss
            loss /= 3

            loss.backward()
            optimizer.step()

            # tqdm_train.set_description("[epoch: %3d/%3d]"
            #                            "[Total loss: %.4f, L1 Loss: %.4f, L2 Loss: %.4f, BCE Loss: %.4f]"
            #                            % (epoch + 1, epoch_num,
            #                               loss.item(), l1_loss.item(), l2_loss.item(), side_loss.item())
            #                            )
            tqdm_train.set_postfix({
                'Total loss': f'{loss.item():.4f}',
                'L1 Loss': f'{l1_loss.item():.4f}',
                'L2 Loss': f'{l2_loss.item():.4f}',
                'BCE Loss': f'{side_loss.item():.4f}'
            })

            if (batch_idx + 1) % (len(trainloader)) == 0:  # Save model at the end of every epoch
                model_save_name = 'E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(batch_idx + 1).zfill(4) + '.pth'
                model_save_pth = os.path.join(pth_path, model_save_name)
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), model_save_pth)  # parallel
                else:
                    torch.save(model.state_dict(), model_save_pth)  # not parallel
                # ###########################--validate--#####################################
                if hr_vm_list is not None:
                    val_out = validate(args, model, img_id=0)  # only validate the first stack
                    hr_volume = tiff.imread(hr_vm_list[0])  # only read the first stack in the dictionary

                    current_psnr = -np.mean((hr_volume - val_out) ** 2)
                    psnr_list.append(current_psnr)

            if epoch + 1 == epoch_num and (batch_idx + 1) % (len(trainloader)) == 0:
                test_immediate_after_train(args, model, epoch, batch_idx)

    #
    if hr_vm_list is not None:
        with open(os.path.join(pth_path, 'psnr.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(psnr_list)

    print('\033[42;37mAll process finished. All models and tests saved to disk.\033[0m')


# ####################################################################################
def validate(args, model, vm_id=0):
    print('\n\033[1;36m--Validation started -----> \033[0m')
    model.eval()  # validation
    with torch.no_grad():
        # Crop test file into 3D patches for inference
        (patches_name_list, patches_gcoordinate_list,
         noise_vm, vm_mean, input_data_type) = patch_volume_for_test(args, vm_id=vm_id)

        denoise_vm = np.zeros(noise_vm.shape)
        input_vm = np.zeros(noise_vm.shape)

        test_data = Testset(patches_name_list, patches_gcoordinate_list, noise_vm)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        tqdm_validate = tqdm(enumerate(testloader))
        for iteration, (noise_patch, single_coordinate) in tqdm_validate:

            real_A = noise_patch.cuda()
            real_A = Variable(real_A)
            fake_B, _ = model(real_A)

            tqdm_validate.set_description("[Patch %d/%d]" % (iteration + 1, len(testloader)))

            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())

            if (output_image.ndim == 3):  # batchsize 1/more
                (output_patch, raw_patch,
                 stack_start_w, stack_end_w,
                 stack_start_h, stack_end_h,
                 stack_start_s, stack_end_s) = singlebatch_test_save(single_coordinate, output_image, raw_image)

                output_patch = output_patch + vm_mean
                raw_patch = raw_patch + vm_mean

                denoise_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                input_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = raw_patch
            else:
                for idx in range(output_image.shape[0]):
                    (output_patch, raw_patch,
                     stack_start_w, stack_end_w,
                     stack_start_h, stack_end_h,
                     stack_start_s, stack_end_s) = multibatch_test_save(single_coordinate, idx, output_image, raw_image)

                    output_patch = output_patch + vm_mean
                    raw_patch = raw_patch + vm_mean

                    denoise_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                    input_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = raw_patch

        # Stitching finish
        output_vm = denoise_vm.squeeze().astype(np.float32) * args.scale_factor
        del denoise_vm

    print('\n\033[1;36m--Validation finished -----> \033[0m')

    return output_vm


# ####################################################################################
def test_immediate_after_train(args, model, epoch, batch_idx):
    print('\n\033[1;31mStart testing -----> \033[0m')
    # %% read vm list:
    noise_vm_list = list(os.walk(args.dataset_path, topdown=False))[-1][-1]
    noise_vm_list.sort()
    print('\033[1;34mPreparing volume images-----> \033[0m')
    print('--Total volume number -----> ', len(noise_vm_list))
    for vm_i in noise_vm_list:
        print('        ----->', vm_i)

    model.eval()  # validation
    with torch.no_grad():
        for N in range(len(noise_vm_list)):
            # Crop test file into 3D patches for inference
            (patches_name_list, patches_gcoordinate_list,
             noise_vm, vm_mean, input_data_type) = patch_volume_for_test(args, N)

            # prev_time = time.time()
            # time_start = time.time()

            denoise_vm = np.zeros(noise_vm.shape)

            pth_name = 'E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(batch_idx + 1).zfill(4)
            result_file_name = noise_vm_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth', '') + '_output.tif'
            result_name = os.path.join(args.output_path, result_file_name)

            test_data = Testset(patches_name_list, patches_gcoordinate_list, noise_vm)
            testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            tqdm_test = tqdm(enumerate(testloader),
                             total=len(testloader), desc=f'Stack {N+1}/{len(noise_vm_list)} {noise_vm_list[N]}',
                             position=0, leave=True)
            for iteration, (noise_patch, single_coordinate) in tqdm_test:
                # for iteration, (noise_patch, single_coordinate) in enumerate(testloader):

                noise_patch = noise_patch.cuda()
                real_A = noise_patch
                real_A = Variable(real_A)

                fake_B, _ = model(real_A)

                # if iteration % 1 == 0:
                #     tqdm_test.set_description(
                #         f'[Stack {N+1}/{len(noise_vm_list)}, {noise_vm_list[N]}] [Patch {iteration+1}/{len(testloader)}]')

                if (iteration + 1) % len(testloader) == 0:
                    print('\n', end=' ')
                # ##############################################################################################

                output_image = np.squeeze(fake_B.cpu().detach().numpy())
                raw_image = np.squeeze(real_A.cpu().detach().numpy())

                if (output_image.ndim == 3):  # batchsize 1/more
                    (output_patch, raw_patch,
                     stack_start_w, stack_end_w,
                     stack_start_h, stack_end_h,
                     stack_start_s, stack_end_s) = singlebatch_test_save(single_coordinate, output_image, raw_image)

                    output_patch = output_patch + vm_mean
                    raw_patch = raw_patch + vm_mean

                    denoise_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                    # input_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    #     = raw_patch
                else:
                    for idx in range(output_image.shape[0]):
                        (output_patch, raw_patch,
                         stack_start_w, stack_end_w,
                         stack_start_h, stack_end_h,
                         stack_start_s, stack_end_s) = multibatch_test_save(single_coordinate, idx, output_image,
                                                                            raw_image)

                        output_patch = output_patch + vm_mean
                        raw_patch = raw_patch + vm_mean

                        denoise_vm[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5

            # Stitching finish
            # del noise_vm
            output_vm = denoise_vm.squeeze().astype(np.float32) * args.scale_factor
            del denoise_vm

            output_vm = np.clip(output_vm, 0, 65535).astype('float32')
            # Save inference image
            if input_data_type == 'uint16':
                output_vm = np.clip(output_vm, 0, 65535)
                output_vm = output_vm.astype('uint16')
            elif input_data_type == 'int16':
                output_vm = np.clip(output_vm, -32767, 32767)
                output_vm = output_vm.astype('int16')
            else:
                output_vm = output_vm.astype('float32')

            io.imsave(result_name, output_vm, check_contrast=False)

        print('\033[1;31mDenoised result saved in:-----> \033[0m', result_name)
