from __future__ import print_function, division
import sys

sys.path.append('core')
sys.path.append('RAFT')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import kornia as K

from raft import RAFT
from syn_dataset import back_dataloader, fore_dataloader
from utils.syn_utils import generate_refer_grid, generate_kernel_grid, get_parameters, augmentation, save_subset, save_wsl_subset, generate_random_fog, save_tepe_subset
from utils.object_utils import random_position
from utils.utils import manual_remap
from PIL import Image
# exclude extremly large displacements
MAX_FLOW = 400

WSL_WEIGHT = 20

def get_device(device='cuda:0'):
    assert isinstance(device, str)
    num_cuda = torch.cuda.device_count()

    if 'cuda' in device:
        if num_cuda > 0:
            return torch.device(device)
        print('No CUDA')
        device = 'cpu'

    if not torch.backends.mkl.is_available():
        raise NotImplementedError('torch.fft on the cpu requires mkl back-end')
    return torch.device('cpu')


def sequence_loss(teacher_preds, student_preds, flow_gt, out_lier, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    #
    n_predictions = len(teacher_preds)
    tt_loss = 0.0
    ts_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (out_lier[:, 0, ...] >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        t_loss = torch.sum((teacher_preds[i] - flow_gt).abs(), dim=1)
        s_loss = torch.sum((student_preds[i] - flow_gt).abs(), dim=1)
        tt_loss += i_weight * (t_loss)
        ts_loss += i_weight * (s_loss)

    tt_loss = tt_loss.view(-1)[valid.view(-1)]
    ts_loss = ts_loss.view(-1)[valid.view(-1)]

    # wsl_loss = torch.exp(-(ts_loss / tt_loss))

    wsl_loss = torch.exp(-(torch.div(ts_loss,tt_loss)))
    wsl_loss = wsl_loss.mean()

    tt_loss = tt_loss.mean()
    ts_loss = ts_loss.mean()

    total_loss = tt_loss + wsl_loss * WSL_WEIGHT

    tepe = torch.sum((teacher_preds[-1] - flow_gt)**2, dim=1).sqrt().view(-1)[valid.view(-1)].mean()
    sepe = torch.sum((student_preds[-1] - flow_gt)**2, dim=1).sqrt().view(-1)[valid.view(-1)].mean()


    # total_loss = 0.5 * (tt_loss * 3*tepe) + wsl_loss * WSL_WEIGHT
    # total_loss = 0.5 * (tt_loss * 3*tepe) + wsl_loss * WSL_WEIGHT
    # total_loss = 3*tepe + wsl_loss * WSL_WEIGHT

    # print("total_loss: %.2f | t_loss : %.2f | s_loss : %.2f | wsl_loss : %.2f | t_epe : %.2f | s_epe : %.2f" % (total_loss, tt_loss, ts_loss, wsl_loss, tepe, sepe))


    return total_loss, tt_loss, ts_loss, tepe, sepe, wsl_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(args):
    device_str = "cuda:0"
    device = get_device(device_str)

    student = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    student_dir = "/home/kwon/eccv/eccv_dupdate/model/manual_remap_modi/best_raft_chairs.pth"
    print(student_dir)
    student.load_state_dict(torch.load(student_dir, map_location=device), strict=False)
    student.to(device)
    student.eval()
    student.module.freeze_bn()

    teacher = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    teacher_dir = "/home/kwon/eccv/eccv_dupdate/model/manual_remap_modi/14000_raft_sintel_noaug_30000.pth"
    print(teacher_dir)
    teacher.load_state_dict(torch.load(teacher_dir, map_location=device), strict=False)
    teacher.to(device)
    teacher.eval()
    teacher.module.freeze_bn()

    back_loader = back_dataloader()
    fore_loader = fore_dataloader()

    is_first = True
    should_keep_training = True

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    save_tloss_folder = os.path.join(args.save_dir, "best_tloss")
    if not os.path.isdir(save_tloss_folder):
        os.mkdir(save_tloss_folder)

    save_wsl_folder = os.path.join(args.save_dir, "best_wsl")
    if not os.path.isdir(save_wsl_folder):
        os.mkdir(save_wsl_folder)

    save_tepe_folder = os.path.join(args.save_dir, "best_tepe")
    if not os.path.isdir(save_tepe_folder):
        os.mkdir(save_tepe_folder)


    if int(len(os.listdir(save_tloss_folder))) == 0:
        save_ind = 0
    else:
        save_ind = int(len(os.listdir(save_tloss_folder)) - 1)

    if int(len(os.listdir(save_wsl_folder))) == 0:
        save_wsl_ind = 0
    else:
        save_wsl_ind = int(len(os.listdir(save_wsl_folder)) - 1)

    if int(len(os.listdir(save_tepe_folder))) == 0:
        save_tepe_ind = 0
    else:
        save_tepe_ind = int(len(os.listdir(save_tepe_folder)) - 1)

    out_w = 512
    out_h = 384
    kss = 0
    while should_keep_training:
        for i_batch, data_blob in enumerate(zip(back_loader, fore_loader)):

            if (save_ind+1) % 1000 == 0:
                print('%d number image save!' % save_ind)

            if save_ind == 20000:
                print("save 20000 images, finish")
                exit()

            fore_num = np.random.randint(8, 13)
            motion_blur_mask_num = np.random.randint(3, 6)

            source_imgs = data_blob[0].to(device)
            back_img = source_imgs[:1]
            fore_rgbs = source_imgs[1:fore_num+1][:, :3, ...]
            fore_alphas = data_blob[1].to(device)[:fore_num+motion_blur_mask_num]
            total_alphas = random_position(back_img, fore_alphas, device).permute(1, 0, 2, 3)
            back_alpha, fore_alphas, motion_blur_masks = torch.split(total_alphas, [1, fore_num, motion_blur_mask_num], 0)

            motion_blur_mask = torch.sum(motion_blur_masks, dim=0, keepdim=True).clamp(0, 1)
            scale_fac = np.random.uniform(1.5, 3)
            motion_blur_mask = F.interpolate(motion_blur_mask, scale_factor=scale_fac, mode='bilinear', align_corners=True)

            _, _, ah, aw = motion_blur_mask.shape

            sh = np.random.randint(0, ah-584)
            sw = np.random.randint(0, aw-712)
            motion_blur_mask = motion_blur_mask[:, :, sh:sh+584, sw:sw+712]

            b, c, h, w = back_img.shape

            cut_h = (h - out_h) // 2
            cut_w = (w - out_w) // 2

            if is_first:
                reference_grid, _ = generate_refer_grid(back_img, 1)
                reference_grid = reference_grid.permute(0, 3, 1, 2).to(device)[0:1]

                is_first = False

            motion_init = torch.zeros([1, 2], device=device)
            motion_init[:, 0] = motion_init[:, 0] * w // 2 * 0.0
            motion_init[:, 1] = motion_init[:, 1] * h // 2 * 0.0
            center_init = torch.zeros([1, 2], device=device)
            zoom_init = torch.ones([1, 2], device=device)
            rot_init = torch.zeros([1], device=device)

            # motion_init.requires_grad = True
            zoom_init.requires_grad = True
            rot_init.requires_grad = True
            center_init.requires_grad = True

            motion_next, _, zoom_next, rot_next = get_parameters(device)
            motion_next[:, 0] = motion_next[:, 0] * 80
            motion_next[:, 1] = motion_next[:, 1] * 50
            center_next = torch.zeros([1, 2], device=device)

            center_next.requires_grad = True
            motion_next.requires_grad = True
            zoom_next.requires_grad = True

            srt_points = torch.tensor([[-w//2, -h//2], [w//2, -h//2], [-w//2, h//2], [w//2, h//2]], dtype=torch.float32, device=device).unsqueeze(0)
            dst_points = torch.tensor([[-w//2, -h//2], [w//2, -h//2], [-w//2, h//2], [w//2, h//2]], dtype=torch.float32, device=device).unsqueeze(0)
            srt_points = srt_points + (torch.rand_like(srt_points, dtype=torch.float32, device=device) - 0.5) * 50
            dst_points = dst_points + (torch.rand_like(dst_points, dtype=torch.float32, device=device) - 0.5) * 50

            srt_points.requires_grad = True
            dst_points.requires_grad = True

            grid_warping1 = torch.zeros([1, 2, 8, 12], dtype=torch.float32, device=device)
            grid_warping2 = torch.zeros([1, 2, 8, 12], dtype=torch.float32, device=device)

            grid_warping1.requires_grad = True
            grid_warping2.requires_grad = True

            alpha_grid = torch.zeros([fore_num, 2, 8, 12], dtype=torch.float32, device=device)
            alpha_grid.requires_grad = True

            fore_offset = (torch.rand(size=(1, (fore_num)*2, 1, 1), dtype=torch.float32, device=device) - 0.5) * 100
            fore_offset.requires_grad = True

            # real world effects

            color_change = (torch.rand(size=(fore_num+1, 3, 1, 1), dtype=torch.float32, device=device) - 0.5) * 2
            color_change.requires_grad = True

            is_color_change = torch.rand(size=(fore_num+1, 1), dtype=torch.float32, device=device).squeeze() < 0.5


            noise1 = torch.rand(size=(1, 3, out_h, out_w), dtype=torch.float32, device=device) * 0.01
            valid_noise1 = torch.rand(size=(1, 3, out_h, out_w), dtype=torch.float32, device=device) < np.random.uniform(0.00, 0.2)
            valid_noise1 = K.filters.box_blur(valid_noise1 * 1.0, kernel_size=(3, 3))
            noise2 = torch.rand(size=(1, 3, out_h, out_w), dtype=torch.float32, device=device) * 0.01
            valid_noise2 = torch.rand(size=(1, 3, out_h, out_w), dtype=torch.float32, device=device) < np.random.uniform(0.00, 0.2)
            valid_noise2 = K.filters.box_blur(valid_noise2 * 1.0, kernel_size=(3, 3))
            texture_noise = torch.rand(size=(1, 3, 8, 24), dtype=torch.float32, device=device) * 0

            noise1.requires_grad = True
            noise2.requires_grad = True
            texture_noise.requires_grad = True
            valid_noise1.requires_grad = True
            valid_noise2.requires_grad = True

            layer_list = []
            layer_ind = 6
            for j in range(fore_num + 1):
                layer_list.append(torch.tensor([layer_ind], device=device, dtype=torch.float32))
                layer_ind += 6
            layer_scalar = torch.cat(layer_list, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            layer_scalar.requires_grad = True

            print('New Image!')
            best_total_loss = 30
            best_wsl = 0.27
            best_tepe = 2.55

            """ Motion Blur Kernel 구현 """
            k_size = 35
            k_offset = int(np.ceil((np.sqrt(2) - 1) * k_size / 2))
            k_origin_grid = generate_kernel_grid(k_size + 2 * k_offset, 1, device)

            k_sigma_x_bs = torch.rand(size=(1, 1, 1), dtype=torch.float32, device=device) * 4 - 2
            k_sigma_y_bs = torch.rand(size=(1, 1, 1), dtype=torch.float32, device=device) * 4 - 2
            k_angle_bs = torch.rand(size=(1, 1), dtype=torch.float32, device=device).squeeze(1) * 4 - 2

            k_sigma_x_bs.requires_grad = True
            k_sigma_y_bs.requires_grad = True
            k_angle_bs.requires_grad = True

            """ Fog """
            fog = generate_random_fog(back_img, device)
            fog_intensity = torch.rand(size=(1, 4, 4), dtype=torch.float32, device=device) * 1
            fog_color = torch.rand(size=(3, 4, 4), dtype=torch.float32, device=device) * 1

            fog_intensity.requires_grad = True
            fog_color.requires_grad = True

            #### 적용할지 말지
            do_motion_blur = np.random.uniform(0, 1) < 0.5
            do_fog = np.random.uniform(0, 1) < 0.5
            do_noise = np.random.uniform(0, 1) < 0.5
            do_large_rot = np.random.uniform(0, 1) < 0.5
            do_color_change = np.random.uniform(0, 1) < 0.5


            if do_large_rot:
                rot_next = rot_next * 5
            else:
                rot_next = rot_next * 30
            rot_next.requires_grad = True

            for k in range(80):

                ll = (2 - (k/50))

                optimizer = optim.Adam(
                    [
                        {"params": motion_init, "lr": 2 * ll}, # (1,2)
                        {"params": zoom_init, "lr": 0.002 * ll}, # (1,2)
                        {"params": rot_init, "lr": 0.01 * ll}, # (1,2)
                        {"params": motion_next, "lr": 1 * ll}, # (1)
                        {"params": zoom_next, "lr": 0.02 * ll},
                        {"params": rot_next, "lr": 0.1 * ll}, # (1,2)
                        {"params": noise1, "lr": 0.01 * ll},
                        {"params": noise2, "lr": 0.01 * ll},
                        {"params": dst_points, 'lr': 0.1 * ll}, # (1,4,2)
                        {"params": grid_warping1, 'lr': 1 * ll},
                        {"params": grid_warping2, 'lr': 1 * ll },
                        {"params": alpha_grid, 'lr': 0.15 * ll},
                        {"params": fore_offset, 'lr': 0.2 * ll},
                        {"params": k_sigma_x_bs, 'lr': 0.03 * ll},
                        {"params": k_sigma_y_bs, 'lr': 0.03 * ll},
                        {"params": fog_color, 'lr': 0.03 * ll},
                        {"params": k_angle_bs, 'lr': 0.03 * ll},
                        {"params": color_change, 'lr': 0.03 * ll}
                    ],
                    lr=0.0001
                )

                update_grid = reference_grid.clone().detach()
                update_back_rgbs = back_img[:, :3, ...].clone().detach()
                update_back_alphas = back_img[:, -1:, ...].clone().detach()
                update_other_rgbs = fore_rgbs.clone().detach()
                update_other_alphas = fore_alphas.clone().detach()

                if do_color_change: # color change를 걸어줌
                    temp_rgb = []
                    temp_alpha = []

                    if is_color_change[0]:
                        temp_rgb.append(update_back_rgbs * torch.sigmoid(color_change[:1, ...]))
                    else:
                        temp_rgb.append(update_back_rgbs)
                    temp_alpha.append(update_back_alphas)
                    for i in range(fore_num):
                        if is_color_change[i+1]:
                            temp_rgb.append(update_other_rgbs[i:i + 1, :3, ...] * torch.sigmoid(color_change[i + 1:i+2, ...]))
                        else:
                            temp_rgb.append(update_other_rgbs[i:i + 1, :3, ...])
                        temp_alpha.append(update_other_alphas[i:i + 1, :1, ...])
                else:
                    temp_rgb = []
                    temp_alpha = []
                    temp_rgb.append(update_back_rgbs)
                    temp_alpha.append(update_back_alphas)
                    for i in range(fore_num):
                        temp_rgb.append(update_other_rgbs[i:i + 1, :3, ...])
                        temp_alpha.append(update_other_alphas[i:i + 1, :1, ...])

                update_rgbs = torch.cat(temp_rgb, dim=1)
                update_alphas = torch.cat(temp_alpha, dim=1)
                update_feature = torch.cat([update_rgbs, update_alphas], dim=1)

                amat_init = K.geometry.transform.get_affine_matrix2d(motion_init, center_init, zoom_init, rot_init)
                t_update_grid1 = torch.cat([update_grid, torch.ones([b, 1, h, w], device=device)], dim=1)  # temp update grid
                w_update_grid1 = torch.matmul(amat_init, t_update_grid1.view(b, 3, -1)).view(b, 3, h, w)[:, :2, ...]  # warped update grid
                u_grid_warping1 = K.geometry.resize(grid_warping1, size=(h, w), interpolation='bilinear', align_corners=False)  # upscaled update grid
                wu_update_grid1 = w_update_grid1 + u_grid_warping1  # warped + upscaled update grid
                wu_update_grid1_x = wu_update_grid1[:, 0, ...] / (w / 2)
                wu_update_grid1_y = wu_update_grid1[:, 1, ...] / (h / 2)

                next_update_features = manual_remap(update_feature, torch.stack([wu_update_grid1_x, wu_update_grid1_y], dim=-1))

                amat_next = K.geometry.transform.get_affine_matrix2d(motion_next, center_next, zoom_next, rot_next)
                pmat_next = K.geometry.get_perspective_transform(srt_points, dst_points) # perspective matrix
                apmat_next = torch.matmul(amat_next[0], pmat_next[0]).unsqueeze(0) # affine + perspective matrix

                """
                각 foreground 마다 가져오고싶은 grid를 다시 설정해줌
                """
                base_grid, fore_grids = torch.split(update_grid.repeat(1, fore_num + 1, 1, 1), [2, fore_num * 2], dim=1)
                u_fore_grid = fore_grids + fore_offset  # updated foreground grid

                u_grid_list = []
                next_rgb_list = []
                next_alpha_list = []
                u_grid_list.append(torch.cat([base_grid, torch.ones([1, 1, h, w], device=device)], dim=1))
                next_rgb_list.append(next_update_features[:, :3, ...])
                next_alpha_list.append(next_update_features[:, 3 * (fore_num + 1):3 * (fore_num + 1) + 1, ...])
                for j in range(fore_num):
                    u_grid_list.append(torch.cat([u_fore_grid[:, j * 2:j * 2 + 2, ...], torch.ones([1, 1, h, w], device=device)], dim=1))
                    next_rgb_list.append(next_update_features[:, (j + 1) * 3:(j + 1) * 3 + 3, ...])
                    next_alpha_list.append(next_update_features[:, 3 * (fore_num + 1) + 1 + j:3 * (fore_num + 1) + 1 + j + 1, ...])

                tb_grid = torch.cat(u_grid_list, dim=0)  # temp batch grid, channel을 batch로 바꿈

                w_update_grid2 = torch.matmul(apmat_next, tb_grid.view(fore_num + 1, 3, -1)).view(fore_num + 1, 3, h, w)[:, :2, ...]  # warped update grid
                u_grid_warping2 = K.geometry.resize(grid_warping2, size=(h, w), interpolation='bilinear', align_corners=False)  # upscaled update grid
                wu_update_grid2 = w_update_grid2 + u_grid_warping2  # warped + upscaled update grid
                wu_update_grid2_x = wu_update_grid2[:, 0, ...] / (w / 2)
                wu_update_grid2_y = wu_update_grid2[:, 1, ...] / (h / 2)

                next_rgbs = torch.cat(next_rgb_list, dim=0)
                next_alphas = torch.cat(next_alpha_list, dim=0)
                next_alpha_first = next_alphas[:1, :, ...] + (1 - next_alphas[:1, :, ...])
                next_alpha_rest = next_alphas[1:, :, ...]

                next_alphas = torch.cat([next_alpha_first, next_alpha_rest], dim=0)
                next_rgba = torch.cat([next_rgbs, next_alphas], dim=1)

                n_object_alphas = next_alphas[:, :, ...] * layer_scalar  # 높은 layer에 더 큰 값을 부여함
                n_object_temp = (n_object_alphas - n_object_alphas.amax(0).unsqueeze(0))
                n_object_weight = (torch.exp(n_object_temp) / torch.exp(n_object_temp).sum(0).unsqueeze(0))  # 각 layer마다 weight를 부여함

                refer_rgba = manual_remap(next_rgba, torch.stack([wu_update_grid2_x, wu_update_grid2_y], dim=-1))
                flows = wu_update_grid2 - update_grid

                refer_alphas = refer_rgba[:, -1:, ...]
                outlier = refer_rgba[:1, -1:, ...].clone().detach()[:, :, cut_h:-cut_h, cut_w:-cut_w]

                refer_alpha_first = refer_alphas[:1, :, ...] + (1 - refer_alphas[:1, :, ...])
                refer_alpha_rest = refer_alphas[1:, :, ...]
                refer_alphas = torch.cat([refer_alpha_first, refer_alpha_rest], dim=0)

                object_alphas = refer_alphas * layer_scalar  # 높은 layer에 더 큰 값을 부여함
                object_temp = (object_alphas - object_alphas.amax(0).unsqueeze(0))
                object_weight = (torch.exp(object_temp) / torch.exp(object_temp).sum(0).unsqueeze(0))  # 각 layer마다 weight를 부여함

                refer_rgb = torch.sum(object_weight * refer_rgba[:, :3, ...], dim=0, keepdim=True)
                next_rgb = torch.sum(n_object_weight * next_rgbs[:, :3, ...], dim=0, keepdim=True)
                flow = torch.sum(object_weight * flows, dim=0, keepdim=True)

                """ Motion Blur! """
                if do_motion_blur:
                    k_sigma_x_as = torch.sigmoid(k_sigma_x_bs) * 8 + 3
                    k_sigma_y_as = torch.sigmoid(k_sigma_y_bs) * 1 + 1

                    k_angle_as = torch.sigmoid(k_angle_bs) * 90

                    k_rotated_grid = K.geometry.rotate(k_origin_grid.permute(0, 3, 1, 2), k_angle_as).permute(0, 2, 3, 1)[:,k_offset:-k_offset, k_offset:-k_offset, :]
                    gauss_x = - k_rotated_grid[..., 0] ** 2 / (2 * k_sigma_x_as ** 2)
                    gauss_y = - k_rotated_grid[..., 1] ** 2 / (2 * k_sigma_y_as ** 2)
                    gauss_kernel = torch.div(torch.exp(gauss_x + gauss_y), torch.abs(2 * np.pi * k_sigma_x_as * k_sigma_y_as))
                    gauss_kernel = torch.div(gauss_kernel, gauss_kernel.sum(1).sum(1)[:, None, None])
                    gauss_kernel = gauss_kernel.unsqueeze(1)
                    gauss_kernel = gauss_kernel.repeat(1, 4, 1, 1)
                    gauss_kernel = gauss_kernel.reshape(-1, 1, k_size, k_size)

                    blur_refer_rgb = refer_rgb
                    blur_next_rgb = next_rgb

                    blur_refer_rgba = torch.cat([blur_refer_rgb, motion_blur_mask], dim=1)
                    blur_next_rgba = torch.cat([blur_next_rgb, motion_blur_mask], dim=1)

                    b1, c1, h1, w1 = blur_refer_rgba.shape

                    blur_refer_rgba = blur_refer_rgba.view(-1, gauss_kernel.size(0), blur_refer_rgba.size(-2), blur_refer_rgba.size(-1))
                    blur_refer_rgba = F.conv2d(blur_refer_rgba, gauss_kernel, groups=gauss_kernel.size(0), padding='same', stride=1)
                    blur_refer_rgba = blur_refer_rgba.view(b1, c1, h1, w1)

                    blur_next_rgba = blur_next_rgba.view(-1, gauss_kernel.size(0), blur_next_rgba.size(-2), blur_next_rgba.size(-1))
                    blur_next_rgba = F.conv2d(blur_next_rgba, gauss_kernel, groups=gauss_kernel.size(0), padding='same', stride=1)
                    blur_next_rgba = blur_next_rgba.view(b1, c1, h1, w1)

                    refer_rgb = (refer_rgb * (1 - blur_refer_rgba[:, -1:, ...]) + blur_refer_rgba[:, :3, ...] * blur_refer_rgba[:, -1:, ...]).clamp(0, 1)
                    next_rgb = (next_rgb * (1 - blur_next_rgba[:, -1:, ...]) + blur_next_rgba[:, :3, ...] * blur_next_rgba[:, -1:, ...]).clamp(0, 1)

                """ Fog! """
                if do_fog:
                    refer_rgb = (refer_rgb * (1 - fog) + fog * torch.sigmoid(K.geometry.resize(fog_color, size=(h, w), interpolation='bilinear',align_corners=False))).clamp(0, 1)
                    next_rgb = (next_rgb * (1 - fog) + fog * torch.sigmoid(K.geometry.resize(fog_color, size=(h, w), interpolation='bilinear', align_corners=False))).clamp(0, 1)

                refer_rgb = refer_rgb[:, :, cut_h:-cut_h, cut_w:-cut_w]
                next_rgb = next_rgb[:, :, cut_h:-cut_h, cut_w:-cut_w]
                flow = flow[:, :, cut_h:-cut_h, cut_w:-cut_w]

                if do_noise:
                    pre_input1 = (refer_rgb + noise1*valid_noise1).clamp(0, 1.0)
                    pre_input2 = (next_rgb + noise2*valid_noise2).clamp(0, 1.0)
                else:
                    pre_input1 = refer_rgb.clamp(0, 1.0)
                    pre_input2 = next_rgb.clamp(0, 1.0)

                pre_gt_flow = flow

                # save_im1 = Image.fromarray(
                #     np.asarray(pre_input1[k].permute(1, 2, 0).detach().cpu() * 255.0, dtype=np.uint8), "RGB")
                # save_im2 = Image.fromarray(
                #     np.asarray(pre_input2[k].permute(1, 2, 0).detach().cpu() * 255.0, dtype=np.uint8), "RGB")
                # save_flo = np.asarray(pre_gt_flow[k].permute(1, 2, 0).detach().cpu() * 64, dtype=np.int16)
                #
                # refer_name = "./check" + "/%02d_0.jpg" % kss
                # next_name = "./check" + "/%02d_1.jpg" % kss
                # flow_name = "./check" + "/%02d_flow" % kss
                #
                # save_im1.save(refer_name)
                # save_im2.save(next_name)
                # np.save(flow_name, save_flo)
                #
                # kss +=1
                # break


                input1, input2, gt_flow, loss_measure_outlier = augmentation(pre_input1, pre_input2, pre_gt_flow, outlier, batch_size=6)

                input1 = input1 * 255.0
                input2 = input2 * 255.0

                student_flow = student(input1, input2, iters=args.iters)
                teacher_flow = teacher(input1, input2, iters=args.iters)

                loss, tt_loss, ts_loss, tepe, sepe, wsl_loss = sequence_loss(teacher_flow, student_flow, gt_flow, loss_measure_outlier)
                grid_loss = F.relu(wu_update_grid2[:, :, :-1, ...] - wu_update_grid2[:, :, 1:, :]).mean() + F.relu(wu_update_grid2[:, :, :, :-1] - wu_update_grid2[:, :, :, 1:]).mean()

                if do_noise:
                    noise_regular = (noise1 * valid_noise1).abs().mean() + (noise2 * valid_noise2).abs().mean()
                    other_loss = grid_loss + noise_regular
                else:
                    other_loss = grid_loss

                loss = loss + other_loss

                if k < 2:
                    if tt_loss > 250:
                        break

                if tt_loss <= best_total_loss:
                    best_total_loss = tt_loss.detach().cpu()
                    save_input1 = pre_input1.detach().cpu()
                    save_input2 = pre_input2.detach().cpu()
                    save_gt_flow = pre_gt_flow.detach().cpu()
                    save_outlier = outlier.repeat(1,3,1,1).detach().cpu()
                    save_total_loss = tt_loss.detach().cpu()

                if tt_loss < 30 and wsl_loss < best_wsl:
                    best_wsl = wsl_loss
                    save_wsl_input1 = pre_input1.detach().cpu()
                    save_wsl_input2 = pre_input2.detach().cpu()
                    save_wsl_gt_flow = pre_gt_flow.detach().cpu()
                    save_outlier = outlier.repeat(1,3,1,1).detach().cpu()
                    save_wsl_loss = wsl_loss.detach().cpu()

                if tepe <= best_tepe:
                    best_tepe = tepe
                    save_tepe_input1 = pre_input1.detach().cpu()
                    save_tepe_input2 = pre_input2.detach().cpu()
                    save_tepe_gt_flow = pre_gt_flow.detach().cpu()
                    save_outlier = outlier.repeat(1,3,1,1).detach().cpu()
                    save_tepe_loss = tepe.detach().cpu()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                del optimizer, loss

            if best_total_loss < 25:
                save_subset(save_input1*255.0, save_input2*255.0, save_outlier * 255.0, save_gt_flow, save_total_loss, save_ind, args)
                save_ind += 1

            if best_wsl < 0.27:
                save_wsl_subset(save_wsl_input1*255.0, save_wsl_input2*255.0, save_outlier * 255.0, save_wsl_gt_flow, save_wsl_loss, save_wsl_ind, args)
                save_wsl_ind += 1

            if best_tepe < 2.55:
                save_tepe_subset(save_tepe_input1 * 255.0, save_tepe_input2 * 255.0, save_outlier * 255.0, save_tepe_gt_flow, save_tepe_loss,
                                save_tepe_ind, args)
                save_tepe_ind += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-ours', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--data_num', type=int, default=20000)
    parser.add_argument('--save_dir', type=str, default="mode_origin")
    parser.add_argument('--teacher_dir', type=str, default="")

    args = parser.parse_args()

    train(args)