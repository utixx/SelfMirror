import datetime
import os
import time

import numpy as np
import torch
from skimage import io
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_process import testset, singlebatch_test_save, multibatch_test_save, test_preprocess_chooseOne


def read_imglist(args):
    im_folder = args.datasets_path
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    print('\033[1;31mStacks for processing -----> \033[0m')
    print('Total stack number -----> ', len(img_list))
    for img in img_list:
        print(img)
    return img_list


#
def read_modellist(args):
    model_path = args.pth_dir + '//' + args.denoise_model
    model_list = list(os.walk(model_path, topdown=False))[-1][-1]
    model_list.sort()

    # calculate the number of model file
    count_pth = 0
    for i in range(len(model_list)):
        aaa = model_list[i]
        if '.pth' in aaa:
            count_pth = count_pth + 1
    args.model_list = model_list
    args.model_list_length = count_pth


#
def test(args, model):
    """
    Pytorch testing workflow
    """
    pth_count = 0
    for pth_index in range(len(args.model_list)):
        aaa = args.model_list[pth_index]
        if '.pth' in aaa:
            pth_count = pth_count + 1
            pth_name = args.model_list[pth_index]
            output_path_name = args.output_path + '//' + pth_name.replace('.pth', '')
            if not os.path.exists(output_path_name):
                os.mkdir(output_path_name)

            # load model
            model_name = args.pth_dir + '//' + args.denoise_model + '//' + pth_name
            print(model_name)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(torch.load(model_name))  # parallel
                model.eval()
            else:
                model.load_state_dict(torch.load(model_name))  # not parallel
                model.eval()
            model.cuda()
            # args.print_img_name = False

            # test all stacks
            for N in range(len(args.img_list)):
                name_list, noise_img, coordinate_list, test_im_name, img_mean, input_data_type = test_preprocess_chooseOne(args, N)
                # print(len(name_list))
                prev_time = time.time()
                time_start = time.time()
                denoise_img = np.zeros(noise_img.shape)
                input_img = np.zeros(noise_img.shape)

                test_data = testset(name_list, coordinate_list, noise_img)
                testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.cuda()
                    real_A = noise_patch

                    real_A = Variable(real_A)
                    fake_B = model(real_A)

                    # Determine approximate time left
                    batches_done = iteration
                    batches_left = 1 * len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    time_left = datetime.timedelta(seconds=time_left_seconds)
                    prev_time = time.time()

                    if iteration % 1 == 0:
                        time_end = time.time()
                        time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                        print(
                            '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                            % (
                                pth_count,
                                args.model_list_length,
                                pth_name,
                                N + 1,
                                len(args.img_list),
                                args.img_list[N],
                                iteration + 1,
                                len(testloader),
                                time_cost,
                                time_left_seconds
                            ), end=' ')

                    if (iteration + 1) % len(testloader) == 0:
                        print('\n', end=' ')

                    # Enhanced sub-stacks are sequentially output from the network
                    output_image = np.squeeze(fake_B.cpu().detach().numpy())
                    raw_image = np.squeeze(real_A.cpu().detach().numpy())
                    if (output_image.ndim == 3):
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
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w] \
                                = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5

                            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w] \
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
                del denoise_img

                # Save inference image
                if args.save_test_images_per_epoch:
                    # if input_data_type == 'uint16':
                    #     output_img = np.clip(output_img, 0, 65535)
                    #     output_img = output_img.astype('uint16')
                    #
                    # elif input_data_type == 'int16':
                    #     output_img = np.clip(output_img, -32767, 32767)
                    #     output_img = output_img.astype('int16')
                    #
                    # else:
                    #     output_img = output_img.astype('int32')

                    output_img = output_img.astype(input_data_type)

                    output_img[output_img<0] = 0

                    result_name = output_path_name + '//' + args.img_list[N].replace('.tif', '') + '_' + pth_name.replace(
                                '.pth', '') + '_' + str(input_data_type) + '.tif'
                    io.imsave(result_name, output_img, check_contrast=False)

                # if pth_count == args.model_list_length:
                #     if args.colab_display:
                #         args.result_display = output_path_name + '//' + args.img_list[N].replace('.tif',
                #                                                                                  '') + '_' + pth_name.replace(
                #             '.pth', '') + '_output.tif'

    print('Testing finished. All results saved to disk.')
