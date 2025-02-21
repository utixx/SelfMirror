import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import tifffile as tiff
from skimage import io
import math
import datetime
import time


# ####################################################################################
class Testset(Dataset):
    #
    def __init__(self, patches_name_list, patches_gcoordinate_list, noise_vm):
        self.patches_name_list = patches_name_list
        self.patches_gcoordinate_list = patches_gcoordinate_list
        self.noise_vm = noise_vm

    def __getitem__(self, index):

        one_patch_gcoordinate = self.patches_gcoordinate_list[self.patches_name_list[index]]

        init_x = one_patch_gcoordinate['init_x']
        end_x = one_patch_gcoordinate['end_x']
        init_y = one_patch_gcoordinate['init_y']
        end_y = one_patch_gcoordinate['end_y']
        init_z = one_patch_gcoordinate['init_z']
        end_z = one_patch_gcoordinate['end_z']

        noise_patch = self.noise_vm[init_z:end_z, init_y:end_y, init_x:end_x]
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))

        return noise_patch, one_patch_gcoordinate

    def __len__(self):
        return len(self.patches_name_list)


# ####################################################################################
def patch_volume_for_test(args, vm_id=0):
    #
    patch_x, patch_y, patch_z = args.patch_x, args.patch_y, args.patch_z
    # gap_x, gap_y, gap_z = args.gap_x, args.gap_y, args.gap_z
    gap_x = int(args.patch_x * (1 - args.overlap_factor_for_test))  # patch gap in x
    gap_y = int(args.patch_y * (1 - args.overlap_factor_for_test))  # patch gap in y
    gap_z = int(args.patch_z * (1 - args.overlap_factor_for_test))  # patch gap in z

    cut_x, cut_y,  cut_z = (patch_x - gap_x) / 2, (patch_y - gap_y) / 2, (patch_z - gap_z) / 2
    assert cut_x >= 0 and cut_y >= 0 and cut_z >= 0, "test cut size is negative!"

    dataset_path, dataset_name = args.dataset_path, args.dataset_name
    select_z_for_test = args.select_z_for_test
    scale_factor = args.scale_factor

    patches_name_list = []
    patches_gcoordinate_list = {}

    noise_vm_list = list(os.walk(dataset_path, topdown=False))[-1][-1]
    noise_vm_list.sort()
    noise_vm_name = noise_vm_list[vm_id]
    noise_vm_path = os.path.join(dataset_path, noise_vm_name)
    noise_vm = tiff.imread(noise_vm_path)

    if noise_vm.shape[0] > select_z_for_test:     # 截断防止空间消耗过大
        noise_vm = noise_vm[0:select_z_for_test, :, :]

    input_data_type = noise_vm.dtype

    vm_mean = noise_vm.mean()
    noise_vm = noise_vm.astype(np.float32) / scale_factor
    noise_vm = noise_vm - vm_mean

    whole_x = noise_vm.shape[2]
    whole_y = noise_vm.shape[1]
    whole_z = noise_vm.shape[0]

    num_x = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_y = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_z = math.ceil((whole_z - patch_z + gap_z) / gap_z)

    for z in range(0, num_z):
        for y in range(0, num_y):
            for x in range(0, num_x):
                one_patch_gcoordinate = {'init_x': 0, 'end_x': 0, 'init_y': 0, 'end_y': 0, 'init_z': 0, 'end_z': 0}

                if x != (num_x - 1):
                    init_x = gap_x * x
                    end_x = gap_x * x + patch_x
                elif x == (num_x - 1):
                    init_x = whole_x - patch_x
                    end_x = whole_x

                if y != (num_y - 1):
                    init_y = gap_y * y
                    end_y = gap_y * y + patch_y
                elif y == (num_y - 1):
                    init_y = whole_y - patch_y
                    end_y = whole_y

                if z != (num_z - 1):
                    init_z = gap_z * z
                    end_z = gap_z * z + patch_z
                elif z == (num_z - 1):
                    init_z = whole_z - patch_z
                    end_z = whole_z

                one_patch_gcoordinate['init_x'] = init_x
                one_patch_gcoordinate['end_x'] = end_x
                one_patch_gcoordinate['init_y'] = init_y
                one_patch_gcoordinate['end_y'] = end_y
                one_patch_gcoordinate['init_z'] = init_z
                one_patch_gcoordinate['end_z'] = end_z

                if x == 0:
                    one_patch_gcoordinate['stack_start_w'] = x * gap_x
                    one_patch_gcoordinate['stack_end_w'] = x * gap_x + patch_x - cut_x
                    one_patch_gcoordinate['patch_start_w'] = 0
                    one_patch_gcoordinate['patch_end_w'] = patch_x - cut_x
                elif x == num_x - 1:
                    one_patch_gcoordinate['stack_start_w'] = whole_x - patch_x + cut_x
                    one_patch_gcoordinate['stack_end_w'] = whole_x
                    one_patch_gcoordinate['patch_start_w'] = cut_x
                    one_patch_gcoordinate['patch_end_w'] = patch_x
                else:
                    one_patch_gcoordinate['stack_start_w'] = x * gap_x + cut_x
                    one_patch_gcoordinate['stack_end_w'] = x * gap_x + patch_x - cut_x
                    one_patch_gcoordinate['patch_start_w'] = cut_x
                    one_patch_gcoordinate['patch_end_w'] = patch_x - cut_x

                if y == 0:
                    one_patch_gcoordinate['stack_start_h'] = y * gap_y
                    one_patch_gcoordinate['stack_end_h'] = y * gap_y + patch_y - cut_y
                    one_patch_gcoordinate['patch_start_h'] = 0
                    one_patch_gcoordinate['patch_end_h'] = patch_y - cut_y
                elif y == num_y - 1:
                    one_patch_gcoordinate['stack_start_h'] = whole_y - patch_y + cut_y
                    one_patch_gcoordinate['stack_end_h'] = whole_y
                    one_patch_gcoordinate['patch_start_h'] = cut_y
                    one_patch_gcoordinate['patch_end_h'] = patch_y
                else:
                    one_patch_gcoordinate['stack_start_h'] = y * gap_y + cut_y
                    one_patch_gcoordinate['stack_end_h'] = y * gap_y + patch_y - cut_y
                    one_patch_gcoordinate['patch_start_h'] = cut_y
                    one_patch_gcoordinate['patch_end_h'] = patch_y - cut_y

                if z == 0:
                    one_patch_gcoordinate['stack_start_s'] = z * gap_z
                    one_patch_gcoordinate['stack_end_s'] = z * gap_z + patch_z - cut_z
                    one_patch_gcoordinate['patch_start_s'] = 0
                    one_patch_gcoordinate['patch_end_s'] = patch_z - cut_z
                elif z == num_z - 1:
                    one_patch_gcoordinate['stack_start_s'] = whole_z - patch_z + cut_z
                    one_patch_gcoordinate['stack_end_s'] = whole_z
                    one_patch_gcoordinate['patch_start_s'] = cut_z
                    one_patch_gcoordinate['patch_end_s'] = patch_z
                else:
                    one_patch_gcoordinate['stack_start_s'] = z * gap_z + cut_z
                    one_patch_gcoordinate['stack_end_s'] = z * gap_z + patch_z - cut_z
                    one_patch_gcoordinate['patch_start_s'] = cut_z
                    one_patch_gcoordinate['patch_end_s'] = patch_z - cut_z

                # noise_patch1 = noise_vm[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = dataset_name + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                patches_name_list.append(patch_name)
                # print(' one_patch_gcoordinate -----> ',one_patch_gcoordinate)
                patches_gcoordinate_list[patch_name] = one_patch_gcoordinate  # {patch_name: one_patch_gcoordinate_{}}

    return patches_name_list, patches_gcoordinate_list, noise_vm, vm_mean, input_data_type


# ####################################################################################
def test(args, model):
    noise_vm_list = args.noise_vm_list
    pth_dir, denoise_model, model_list = args.pth_store_dir, args.denoise_models, args.model_list
    batch_size, scale_factor = args.batch_size, args.scale_factor
    output_path = args.output_path

    for pth_index in range(len(model_list)):
        model_i = model_list[pth_index]
        if '.pth' in model_i:
            pth_name = model_list[pth_index]
            # output_vm_path = os.path.join(output_path, pth_name.replace('.pth', ''))

            # if not os.path.exists(output_vm_path):
            #     os.mkdir(output_vm_path)

            # load model
            model_name = os.path.join(pth_dir, denoise_model, pth_name)

            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(torch.load(model_name))  # parallel
                model.eval()
            else:
                model.load_state_dict(torch.load(model_name))  # not parallel
                model.eval()
            model.cuda()

            # test all stacks
            for N in range(len(noise_vm_list)):
                #
                (patches_name_list, patches_gcoordinate_list,
                 noise_vm, vm_mean, input_data_type) = patch_volume_for_test(args, N)

                prev_time = time.time()
                time_start = time.time()

                denoise_vm = np.zeros(noise_vm.shape)

                result_file_name = noise_vm_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth', '') + '_output.tif'
                result_name = os.path.join(output_path, result_file_name)
                # print(os.getcwd())
                # print(result_name)

                test_data = Testset(patches_name_list, patches_gcoordinate_list, noise_vm)
                testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    for iteration, (noise_patch, one_patch_gcoordinate) in enumerate(testloader):

                        noise_patch = noise_patch.cuda()
                        real_A = noise_patch
                        real_A = Variable(real_A)

                        fake_B = model(real_A)

                        # ##############################################################################################
                        # Determine approximate time left
                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(batches_left * (time.time() - prev_time))
                        time_left = datetime.timedelta(seconds=time_left_seconds)
                        prev_time = time.time()
                        # ##############################################################################################
                        if iteration % 1 == 0:
                            time_end = time.time()
                            time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                            print(
                                '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]   '
                                % (
                                    pth_index + 1, len(model_list), pth_name,
                                    N + 1, len(noise_vm_list), noise_vm_list[N],
                                    iteration + 1, len(testloader),
                                    time_cost, time_left_seconds
                                ), end=' ')

                        if (iteration + 1) % len(testloader) == 0:
                            print('\n', end=' ')
                        # ##############################################################################################
                        output_patch = np.squeeze(fake_B.cpu().detach().numpy())
                        raw_patch = np.squeeze(real_A.cpu().detach().numpy())

                        if output_patch.ndim == 3:
                            (aaaa, bbbb,
                             stack_start_w, stack_end_w,
                             stack_start_h, stack_end_h,
                             stack_start_s, stack_end_s) = singlebatch_test_save(one_patch_gcoordinate, 
                                                                                 output_patch, raw_patch)
                            aaaa = aaaa + vm_mean
                            bbbb = bbbb + vm_mean

                            denoise_vm[stack_start_s:stack_end_s,
                                        stack_start_h:stack_end_h,
                                        stack_start_w:stack_end_w] = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
                        else:
                            turn = output_patch.shape[0]
                            for id in range(turn):  # [volume1, volume2, ...]
                                # print('shape of output_patch -----> ',output_patch.shape)
                                (aaaa, bbbb,
                                 stack_start_w, stack_end_w,
                                 stack_start_h, stack_end_h,
                                 stack_start_s, stack_end_s) = multibatch_test_save(one_patch_gcoordinate, id,
                                                                                    output_patch, raw_patch)
                                aaaa = aaaa + vm_mean
                                bbbb = bbbb + vm_mean

                                denoise_vm[stack_start_s:stack_end_s,
                                            stack_start_h:stack_end_h,
                                            stack_start_w:stack_end_w] = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5

                    del noise_vm
                    output_vm = denoise_vm.squeeze().astype(np.float32) * scale_factor
                    del denoise_vm

                    output_vm = np.clip(output_vm, 0, 65535).astype('int32')
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


# ####################################################################################
def singlebatch_test_save(one_patch_gcoordinate, output_patch, raw_patch):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks (if the batch 
    size equal to 1).

    Args:
        one_patch_gcoordinate : the coordinate dict of the image
        output_patch : the output sub-stack of the network
        raw_patch : the noisy sub-stack
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w = int(one_patch_gcoordinate['stack_start_w'])
    stack_end_w = int(one_patch_gcoordinate['stack_end_w'])
    patch_start_w = int(one_patch_gcoordinate['patch_start_w'])
    patch_end_w = int(one_patch_gcoordinate['patch_end_w'])

    stack_start_h = int(one_patch_gcoordinate['stack_start_h'])
    stack_end_h = int(one_patch_gcoordinate['stack_end_h'])
    patch_start_h = int(one_patch_gcoordinate['patch_start_h'])
    patch_end_h = int(one_patch_gcoordinate['patch_end_h'])

    stack_start_s = int(one_patch_gcoordinate['stack_start_s'])
    stack_end_s = int(one_patch_gcoordinate['stack_end_s'])
    patch_start_s = int(one_patch_gcoordinate['patch_start_s'])
    patch_end_s = int(one_patch_gcoordinate['patch_end_s'])

    output_patch = output_patch[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_patch[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(one_patch_gcoordinate, id, output_patch, raw_patch):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks. (if the batch size larger than 1).

    Args:
        one_patch_gcoordinate : the coordinate dict of the image
        output_patch : the output sub-stack of the network
        raw_patch : the noisy sub-stack
    Returns:
        \
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w_id = one_patch_gcoordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = one_patch_gcoordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = one_patch_gcoordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = one_patch_gcoordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = one_patch_gcoordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = one_patch_gcoordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = one_patch_gcoordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = one_patch_gcoordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = one_patch_gcoordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = one_patch_gcoordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = one_patch_gcoordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = one_patch_gcoordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_patch_id = output_patch[id]
    raw_patch_id = raw_patch[id]
    output_patch = output_patch_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_patch_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s
