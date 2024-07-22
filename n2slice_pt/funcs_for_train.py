import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from data_process import trainset, test_preprocess_chooseOne, testset, multibatch_test_save, singlebatch_test_save

import numpy as np
import os
import tifffile as tif
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
import csv
from tqdm import tqdm


#
def patch_prepare_for_training(args):
    """
    The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Important Fields:
       args.name_list : the coordinates of 3D patch are indexed by the patch name in args.name_list. \n
       args.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack. \n
       args.stack_index : the index of the noisy stacks. \n
       args.noise_im_all : the collection of all noisy stacks. \n
    """
    name_list = []
    coordinate_list = {}
    stack_index = []
    noise_im_all = []
    ind = 0

    print('\033[1;34mImage list for training -----> \033[0m')
    stack_num = len(list(os.walk(args.datasets_path, topdown=False))[-1][-1])
    print('--Total stack number -----> ', stack_num)

    for im_name in list(os.walk(args.datasets_path, topdown=False))[-1][-1]:
        print('---Noise image name -----> ', im_name)

        im_dir = args.datasets_path + '//' + im_name
        noise_im = tif.imread(im_dir)
        # if noise_im.shape[0] > args.select_img_num:  # t序列最大数, 超过就截断
        #     noise_im = noise_im[0:args.select_img_num, :, :]

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_z = noise_im.shape[0]

        print('---Noise image shape -----> ', noise_im.shape)
        # Calculate real args.gap_z
        # 明白了,相当于计算z或者t轴的步长, 来满足我有多少个patches, args.train_datasets_size参数
        w_num = math.floor((whole_x - args.patch_x) / args.gap_x) + 1
        h_num = math.floor((whole_y - args.patch_y) / args.gap_y) + 1
        s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)  # stack_num是有多少个三维图像
        # print(s_num)
        args.gap_z = math.floor((whole_z - args.patch_z * 2) / (s_num - 1))
        print('---real gap_z--->', args.gap_z)

        # # No preprocessing
        # noise_im = noise_im.astype(np.float32) / args.scale_factor
        # Minus mean before training
        noise_im = noise_im.astype(np.float32) / args.scale_factor
        noise_im = noise_im - noise_im.mean()

        noise_im_all.append(noise_im)
        args.patch_z2 = args.patch_z * 2

        for x in range(0, int((whole_y - args.patch_y + args.gap_y) / args.gap_y)):
            for y in range(0, int((whole_x - args.patch_x + args.gap_x) / args.gap_x)):
                for z in range(0, int((whole_z - args.patch_z2 + args.gap_z) / args.gap_z)):
                # for z in range(0, int((whole_z - args.patch_z + args.gap_z) / args.gap_z)):
                    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                    init_h = args.gap_y * x
                    end_h = args.gap_y * x + args.patch_y
                    init_w = args.gap_x * y
                    end_w = args.gap_x * y + args.patch_x
                    init_s = args.gap_z * z
                    # end_s = args.gap_z * z + args.patch_z2
                    end_s = args.gap_z * z + args.patch_z

                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s

                    patch_name = args.datasets_name + '_' + im_name.replace('.tif', '') + '_x' + str(
                        x) + '_y' + str(y) + '_z' + str(z)

                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1

    return name_list, coordinate_list, stack_index, noise_im_all


#
def train(args, model):
    current_psnr = 0.0
    best_psnr = 0.0
    psnr_list = []

    print("\033[1;34mdefine optimizer -----> \033[0m")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print("\033[1;34mdefine loss func -----> \033[0m")
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()
    L2.cuda()
    L1.cuda()

    print("\033[1;34mstart training -----> \033[0m")

    # train_data = trainset(args.name_list, args.coordinate_list, args.noise_im_all, args.stack_index)
    # # print(len(train_data))
    # trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(0, args.epoch_num):
        model.train()
        # load dataset very epoch ? why   好像是因为用了distribute gpu  详见:https://zhuanlan.zhihu.com/p/552476081
        train_data = trainset(args.name_list, args.coordinate_list, args.noise_im_all, args.stack_index)
        # print(len(train_data))
        trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        tqdm_train = tqdm(enumerate(trainloader))
        for batch_idx, (inputs, targets) in tqdm_train:
            optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs_v = Variable(inputs, requires_grad=False)
            targets_v = Variable(targets, requires_grad=False)

            outputs = model(inputs_v)

            l1_loss = L1(outputs, targets_v)
            l2_loss = L2(outputs, targets_v)
            loss = 0.5 * l1_loss + 0.5 * l2_loss

            loss.backward()
            optimizer.step()

            # print()
            tqdm_train.set_description("[epoch: %3d/%3d, batch: %d/%d]"
                                       "[Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f]"
                                       % (epoch + 1, args.epoch_num, batch_idx + 1, len(trainloader),
                                          loss.item(), l1_loss.item(), l2_loss.item()))
            # tqdm_mesg.update()
            # Save model at the end of every epoch

            if (batch_idx + 1) % (len(trainloader)) == 0:
                if args.hr_image is not None:

                    val_out = validate(args, model, img_id=0)  # only validate the first stack

                    hr_image = tif.imread(args.hr_image[0])  # only read the first stack in the dictionary
                    current_psnr = compare_psnr(hr_image, val_out)
                    psnr_list.append(current_psnr)

                    if current_psnr > best_psnr:
                        best_psnr = current_psnr
                        print('checkpoint was saved!!! Best Avg. PSNR: {}; Current PSNR: {}'.format(best_psnr,
                                                                                                    current_psnr))
                        model_save_name = args.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(
                            batch_idx + 1).zfill(4) + '.pth'

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), model_save_name)  # parallel
                        else:
                            torch.save(model.state_dict(), model_save_name)  # not parallel
                else:
                    model_save_name = args.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(
                        batch_idx + 1).zfill(4) + '.pth'
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), model_save_name)  # parallel
                    else:
                        torch.save(model.state_dict(), model_save_name)  # not parallel
        # tqdm_mesg.close()

    # with open(args.pth_path + '//' + 'psnr.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(psnr_list)

    print('\033[42;37mTraining finished. All models saved to disk.\033[0m')


def validate(args, model, img_id=0):
    print('\n\033[1;36m--Validation started -----> \033[0m')
    model.eval()  # validation
    with torch.no_grad():
        name_list, noise_img, coordinate_list, \
            test_im_name, img_mean, input_data_type = test_preprocess_chooseOne(args, img_id=img_id)

        denoise_img = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)

        test_data = testset(name_list, coordinate_list, noise_img)
        testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        tqdm_validate = tqdm(enumerate(testloader))
        for iteration, (noise_patch, single_coordinate) in tqdm_validate:
            # Pre-trained models are loaded into memory and the sub-stacks are directly fed into the model.
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = model(real_A)

            tqdm_validate.set_description("[Patch %d/%d]" % (iteration + 1, len(testloader)))
            tqdm_validate.update()

            # if iteration % 1 == 0:
            #     time_end = time.time()
            #     time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
            #     print(
            #         '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
            #         % (
            #             iteration + 1, len(testloader),
            #             time_cost, time_left_seconds
            #         ), end=' ')
            #
            # if (iteration + 1) % len(testloader) == 0:
            #     print('\n', end=' ')

            # Enhanced sub-stacks are sequentially output from the network
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            if (output_image.ndim == 3):  # batchsize 1/more
                postprocess_turn = 1
            else:
                postprocess_turn = output_image.shape[0]

            # The final enhanced stack can be obtained by stitching all sub-stacks.
            if (postprocess_turn > 1):

                for id in range(postprocess_turn):
                    output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                        single_coordinate, id, output_image, raw_image)

                    output_patch = output_patch + img_mean
                    raw_patch = raw_patch + img_mean

                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                    input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = raw_patch
            else:
                output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                    single_coordinate, output_image, raw_image)

                output_patch = output_patch + img_mean
                raw_patch = raw_patch + img_mean

                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = raw_patch

        # Stitching finish
        output_img = denoise_img.squeeze().astype(np.float32) * args.scale_factor
        output_img[output_img < 0] = 0
        del denoise_img

        return output_img


def compare_psnr(img1, img2):
    psnr_value = 0.0
    for i in range(img1.shape[0]):
        psnr_value += psnr(img1[i], img2[i], data_range=max(img1[i].max(), img2[i].max()))
    psnr_value /= img1.shape[0]  # average

    return psnr_value.item()

