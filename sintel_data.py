from __future__ import print_function, division
import sys

sys.path.append('core')
sys.path.append('RAFT')
sys.path.append('third_party/random_polygon')

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import kornia as K

from raft import RAFT
from syn_dataset import back_dataloader, fore_dataloader
from utils.syn_utils import generate_refer_grid, generate_kernel_grid, get_parameters, augmentation, save_subset, affine_warp, generate_random_fog
from utils.object_utils import random_position
from utils.utils import manual_remap
from third_party.random_polygon.src_generate import polygon_mask_gan

# Exclude extremly large displacements
MAX_FLOW = 400

# Weight of balanced loss 
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

    temp = 0

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
        
        temp += i_weight

    tt_loss = tt_loss.view(-1)[valid.view(-1)]
    ts_loss = ts_loss.view(-1)[valid.view(-1)]

    wsl_loss = torch.exp(-(torch.div(ts_loss,tt_loss)))
    wsl_loss = wsl_loss.mean()

    tt_loss = tt_loss.mean()
    ts_loss = ts_loss.mean()

    total_loss = tt_loss + wsl_loss * WSL_WEIGHT

    tepe = torch.sum((teacher_preds[-1] - flow_gt)**2, dim=1).sqrt().view(-1)[valid.view(-1)].mean()
    sepe = torch.sum((student_preds[-1] - flow_gt)**2, dim=1).sqrt().view(-1)[valid.view(-1)].mean()

    return total_loss, tt_loss, ts_loss, tepe, sepe, wsl_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lr_func(k):
    return 2*(1 - 1 / 240 * k) / 3
    # if k < 30:
    #     return 1 - 3 / 100 * k
    # elif 30 <= k < 60:
    #     return 1 - 3 / 100 * (k-30)
    # elif 60 <= k < 90:
    #     return 1 - 3 / 100 * (k-60)
    # else:
    #     return 1 - 3 / 100 * (k-90)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(args):
   
    if args.data_size == "T":
        out_w = 960
        out_h = 540
        aug_w = 720
        aug_h = 400

        ds = [out_w, out_h, aug_w, aug_h]
    
    else:
        print('Data size is requried!')
        exit()

    device_str = "cuda:0"
    device = get_device(device_str)

    student = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    student_dir = "./models/best_raft_chairs.pth"
    print(student_dir)
    student.load_state_dict(torch.load(student_dir, map_location=device), strict=False)
    student.to(device)
    student.eval()
    student.module.freeze_bn()

    teacher = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    teacher_dir = "./models/14000_raft_sintel_noaug_30000.pth"
    print(teacher_dir)
    teacher.load_state_dict(torch.load(teacher_dir, map_location=device), strict=False)
    teacher.to(device)
    teacher.eval()
    teacher.module.freeze_bn()

    back_loader = back_dataloader(ds)
    fore_loader = fore_dataloader()

    is_first = True
    should_keep_training = True

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(os.path.join(args.save_dir, "best_tloss"))


    save_folder = args.save_dir
    # if not os.path.isdir(save_folder):
    #     os.mkdir(save_folder)
    #     os.mkdir(os.path.join(save_folder, "best_tloss"))

    if int(len(os.listdir(os.path.join(save_folder, "best_tloss")))) == 0:
        save_ind = 0
    else:
        save_ind = int(len(os.listdir(os.path.join(save_folder, "best_tloss"))))

    print(save_ind)
    
    while should_keep_training:
        for _, data_blob in enumerate(zip(back_loader, fore_loader)):

            if (save_ind+1) % 100 == 0:
                print('%d number image save!' % save_ind)

            if save_ind == 2000:
                print("save 2000 images, finish")
                exit()
            
            fore_num = np.random.randint(6, 8)
            motion_blur_mask_num = np.random.randint(2, 4)

            source_imgs = data_blob[0].to(device)
            back_img = source_imgs[:1] # background image
            fore_rgbs = source_imgs[1:fore_num+1][:, :3, ...] # foreground rgb
            fore_alphas = data_blob[1].to(device)[:fore_num+motion_blur_mask_num] # foreground alpha + motion blur alpha
    
            total_alphas = random_position(back_img, fore_alphas, device).permute(1, 0, 2, 3)

            total_mask_list = []
            total_mask_list.append(torch.ones_like(back_img[:, :1, ...]))
            for i in range(fore_num + motion_blur_mask_num):
                rand_h = np.random.randint(200, 360)
                rand_w = np.random.randint(200, 360)
                mask = polygon_mask_gan.get_random_mask(rand_h, rand_w)
                mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                total_mask_list.append(mask)
            
            total_alpha_list = []
            total_alpha_list.append(total_mask_list[0])
            for i in range(1, len(total_mask_list)):
                base_mask = torch.zeros_like(total_alpha_list[0])
                _, _, mh, mw = total_mask_list[i].shape
                sh = np.random.randint(0, base_mask.shape[2]-mh)
                sw = np.random.randint(0, base_mask.shape[3]-mw)
                base_mask[:, :, sh:sh+mh, sw:sw+mw] = total_mask_list[i]
                total_alpha_list.append(base_mask)
            
            total_alphas = torch.cat(total_alpha_list, 0)


            back_alpha, fore_alphas, motion_blur_masks = torch.split(total_alphas, [1, fore_num, motion_blur_mask_num], 0)

            motion_blur_mask = torch.sum(motion_blur_masks, dim=0, keepdim=True).clamp(0, 1)

            b, c, h, w = back_img.shape

            cut_h = (h - ds[1]) // 2
            cut_w = (w - ds[0]) // 2

            if is_first:
                reference_grid, _ = generate_refer_grid(back_img, 1)
                reference_grid = reference_grid.permute(0, 3, 1, 2).to(device)[0:1]
                is_first = False

            motion_init, zoom_init, rot_init = get_parameters(device, init=True)
            zoom_init.requires_grad = True
            rot_init.requires_grad = True
            motion_init.requires_grad = True

            motion_next, zoom_next, rot_next = get_parameters(device)
            motion_next.requires_grad = True
            zoom_next.requires_grad = True
            rot_next.requires_grad = True

            srt_points = torch.tensor([[-w//2, -h//2], [w//2, -h//2], [-w//2, h//2], [w//2, h//2]], dtype=torch.float32, device=device).unsqueeze(0)
            dst_points = torch.tensor([[-w//2, -h//2], [w//2, -h//2], [-w//2, h//2], [w//2, h//2]], dtype=torch.float32, device=device).unsqueeze(0)
            srt_points = srt_points + (torch.rand_like(srt_points, dtype=torch.float32, device=device) - 0.5) * 50
            dst_points = dst_points + (torch.rand_like(dst_points, dtype=torch.float32, device=device) - 0.5) * 50
            srt_points.requires_grad = True
            dst_points.requires_grad = True

            grid_warping1 = (torch.rand([1, 2, 8, 12], dtype=torch.float32, device=device) - 0.5) * 6
            grid_warping2 = (torch.rand([1, 2, 8, 12], dtype=torch.float32, device=device) - 0.5) * 6
            grid_warping1.requires_grad = True
            grid_warping2.requires_grad = True

            fore_offset = (torch.rand(size=(fore_num, 2, 1, 1), dtype=torch.float32, device=device) - 0.5) * 150
            fore_rotation = torch.zeros((fore_num, 2), dtype=torch.float32, device=device)
            fore_theta = (torch.rand((fore_num,)) - 0.5) * np.pi 
            fore_rotation[:,0] = torch.cos(fore_theta)
            fore_rotation[:,1] = torch.sin(fore_theta)
            fore_offset.requires_grad = True
            fore_rotation.requires_grad = True

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

            layer_list = []
            layer_ind = 6
            for j in range(fore_num + 1):
                layer_list.append(torch.tensor([layer_ind], device=device, dtype=torch.float32))
                layer_ind += 6
            layer_scalar = torch.cat(layer_list, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            layer_scalar.requires_grad = True

            print('New Image!')
            best_total_loss = 30

            """ Motion Blur Kernel 구현 """
            k_size = 35
            k_offset = int(np.ceil((np.sqrt(2) - 1) * k_size / 2))
            k_origin_grid = generate_kernel_grid(k_size + 2 * k_offset, 1, device)

            k_sigma_x_bs = torch.rand(size=(1, 1, 1), dtype=torch.float32, device=device) * 4 - 2
            k_sigma_y_bs = torch.rand(size=(1, 1, 1), dtype=torch.float32, device=device) * 4 - 2
            k_angle_bs = torch.zeros((2,), dtype=torch.float32, device=device)
            theta = (torch.rand((1,)) - 0.5) * np.pi 
            k_angle_bs[0] = torch.cos(theta)
            k_angle_bs[1] = torch.sin(theta)

            k_sigma_x_bs.requires_grad = True
            k_sigma_y_bs.requires_grad = True
            k_angle_bs.requires_grad = True

            """ Fog """
            fog = generate_random_fog(back_img, device)
            fog_color = torch.rand(size=(3, 4, 4), dtype=torch.float32, device=device) * 1

            fog_color.requires_grad = True

            #### 적용할지 말지
            do_motion_blur = np.random.uniform(0, 1) < 0.5
            do_fog = np.random.uniform(0, 1) < 0.5
            do_noise = np.random.uniform(0, 1) < 0.5
            do_color_change = np.random.uniform(0, 1) < 0.5
            # do_motion_blur = np.random.uniform(0, 1) < 2
            # do_fog = np.random.uniform(0, 1) < 2
            # do_noise = np.random.uniform(0, 1) < 2
            # do_color_change = np.random.uniform(0, 1) < 2
            rot_next.requires_grad = True

            optimizer = optim.Adam(
                    [
                        {"params": motion_init, "lr": 0.5}, # (1,2)
                        {"params": zoom_init, "lr": 0.002}, # (1,2)
                        {"params": rot_init, "lr": 0.1}, # (1,2)
                        {"params": motion_next, "lr": 0.5}, # (1)
                        {"params": zoom_next, "lr": 0.015},
                        {"params": rot_next, "lr": 0.1}, # (1,2)
                        {"params": noise1, "lr": 0.01},
                        {"params": noise2, "lr": 0.01},
                        {"params": grid_warping1, 'lr': 0.5},
                        {"params": grid_warping2, 'lr': 2},
                        {"params": fore_offset, 'lr': 0.2},
                        {"params": fore_rotation, 'lr': 0.1},
                        {"params": k_sigma_x_bs, 'lr': 0.03},
                        {"params": k_sigma_y_bs, 'lr': 0.03},
                        {"params": fog_color, 'lr': 0.03},
                        {"params": k_angle_bs, 'lr': 0.1},
                        {"params": color_change, 'lr': 0.03}
                    ],
                )
            
            print(k_sigma_x_bs)
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

            for k in range(140):

                # print(rot_next)
                optimizer.zero_grad()

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

                update_rgbs = torch.cat(temp_rgb, dim=0)
                update_alphas = torch.cat(temp_alpha, dim=0)
                update_feature = torch.cat([update_rgbs, update_alphas], dim=1)

                u_grid_warping1 = K.geometry.resize(grid_warping1, size=(h, w), interpolation='bilinear', align_corners=False)  # upscaled update grid
                u_update_grid1 = update_grid + u_grid_warping1  # warped + upscaled update grid
                wu_update_grid1 = affine_warp(u_update_grid1, motion_init, zoom_init, rot_init, device)

                wu_update_grid1 = wu_update_grid1.repeat(fore_num+1, 1, 1, 1)
                wu_update_grid1_x = wu_update_grid1[:, 0, ...] / (w / 2)
                wu_update_grid1_y = wu_update_grid1[:, 1, ...] / (h / 2)

                next_update_features = manual_remap(update_feature, torch.stack([wu_update_grid1_x, wu_update_grid1_y], dim=-1))

                base_grid = update_grid
                u_fore_grid = base_grid.repeat(fore_num, 1, 1, 1)

                fore_rot_mat = torch.zeros((fore_num, 2, 2), dtype=torch.float32, device=device)
                fore_rot_normalized = fore_rotation / torch.sqrt(fore_rotation[:,:1]**2 + fore_rotation[:,1:2] ** 2)

                fore_rot_mat[:, 0, 0] = fore_rot_normalized[:, 0]
                fore_rot_mat[:, 0, 1] = fore_rot_normalized[:, 1]
                fore_rot_mat[:, 1, 0] = -fore_rot_normalized[:, 1]
                fore_rot_mat[:, 1, 1] = fore_rot_normalized[:, 0]

                bbb, ccc, hhh, www = u_fore_grid.shape
                u_fore_grid = torch.bmm(u_fore_grid.view(fore_num, 2, -1).permute(0, 2, 1), fore_rot_mat).permute(0, 2, 1).view(fore_num, 2, hhh, www) + fore_offset
                
                bu_grid = torch.cat([base_grid, u_fore_grid], 0)
                u_grid_warping2 = K.geometry.resize(grid_warping2, size=(h, w), interpolation='bilinear', align_corners=False)  # upscaled update grid
                bu_grid = bu_grid + u_grid_warping2 # warped + upscaled update grid

                grid_loss = F.relu(bu_grid[:, :, :-1, ...] - bu_grid[:, :, 1:, :]).mean() + F.relu(bu_grid[:, :, :, :-1] - bu_grid[:, :, :, 1:]).mean()

                bu_grid = affine_warp(bu_grid, motion_next, zoom_next, rot_next, device, srt_points, dst_points)
                
                wu_update_grid2 = bu_grid
                wu_update_grid2_x = wu_update_grid2[:, 0, ...] / (w / 2)
                wu_update_grid2_y = wu_update_grid2[:, 1, ...] / (h / 2)

                refer_rgba = manual_remap(next_update_features, torch.stack([wu_update_grid2_x, wu_update_grid2_y], dim=-1))
                flows = wu_update_grid2 - update_grid

                n_object_alphas = next_update_features[:, -1:, ...] * layer_scalar  # 높은 layer에 더 큰 값을 부여함
                n_object_temp = (n_object_alphas - n_object_alphas.amax(0).unsqueeze(0))
                n_object_weight = (torch.exp(n_object_temp) / torch.exp(n_object_temp).sum(0).unsqueeze(0))  # 각 layer마다 weight를 부여함

                outlier = refer_rgba[:1, -1:, ...].clone().detach()[:, :, cut_h:-cut_h, cut_w:-cut_w]

                object_alphas = refer_rgba[:, -1:, ...] * layer_scalar  # 높은 layer에 더 큰 값을 부여함
                object_temp = (object_alphas - object_alphas.amax(0).unsqueeze(0))
                object_weight = (torch.exp(object_temp) / torch.exp(object_temp).sum(0).unsqueeze(0))  # 각 layer마다 weight를 부여함

                refer_rgb = torch.sum(object_weight * refer_rgba[:, :3, ...], dim=0, keepdim=True)
                next_rgb = torch.sum(n_object_weight * next_update_features[:, :3, ...], dim=0, keepdim=True)
                flow = torch.sum(object_weight * flows, dim=0, keepdim=True)

                """ Motion Blur! """
                if do_motion_blur:
                    k_sigma_x_as = torch.sigmoid(k_sigma_x_bs) * 8 + 3
                    k_sigma_y_as = k_sigma_x_as * 0.4

                    k_angle_as = k_angle_bs

                    rot_mat = torch.zeros((2, 2), dtype=torch.float32, device=device)
                    rot_normalized = k_angle_as / torch.sqrt(k_angle_as[0]**2 + k_angle_as[1] ** 2)
                    rot_mat[0, 0] = rot_normalized[0]
                    rot_mat[0, 1] = rot_normalized[1]
                    rot_mat[1, 0] = -rot_normalized[1]
                    rot_mat[1, 1] = rot_normalized[0]

                    k_rotated_grid = torch.matmul(k_origin_grid, rot_mat)[:,k_offset:-k_offset, k_offset:-k_offset, :]
                    
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
                input1, input2, gt_flow, loss_measure_outlier = augmentation(pre_input1, pre_input2, pre_gt_flow, outlier, ds, batch_size=2)
                
                flow_init = (F.interpolate(gt_flow, scale_factor=1/8, mode='bilinear') / 8) * 0.5
                input1 = input1 
                input2 = input2

                student_flow = student(input1, input2, flow_init=flow_init, iters=args.iters)
                teacher_flow = teacher(input1, input2, flow_init=flow_init, iters=args.iters)

                loss, tt_loss, _, tepe, _, _ = sequence_loss(teacher_flow, student_flow, gt_flow, loss_measure_outlier)

                if do_noise:
                    noise_regular = (noise1 * valid_noise1).abs().mean() + (noise2 * valid_noise2).abs().mean()
                    other_loss = grid_loss + noise_regular
                else:
                    other_loss = grid_loss

                loss = loss + other_loss
                print("tt_loss: %.2f, t_epe: %.2f" % (tt_loss, tepe))

                if k < 2:
                    if tt_loss > 500:
                        break

                if tt_loss <= best_total_loss:
                    best_total_loss = tt_loss.detach().cpu()
                    save_input1 = pre_input1.detach().cpu()
                    save_input2 = pre_input2.detach().cpu()
                    save_gt_flow = pre_gt_flow.detach().cpu()
                    save_outlier = outlier.repeat(1,3,1,1).detach().cpu()
                    save_total_loss = tt_loss.detach().cpu()

                loss.backward()
                optimizer.step()
                scheduler.step()

            if best_total_loss < 12:
                save_subset(save_input1*255.0, save_input2*255.0, save_outlier * 255.0, save_gt_flow, save_total_loss, save_ind, args)
                save_ind += 1              



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-ours', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--data_num', type=int, default=20000)
    parser.add_argument('--save_dir', type=str, default="mode_origin")
    parser.add_argument('--teacher_dir', type=str, default="")

    parser.add_argument('--data_size', default='T')

    args = parser.parse_args()

    train(args)