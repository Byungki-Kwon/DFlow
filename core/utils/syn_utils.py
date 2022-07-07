import torch
import numpy as np
import cv2
from torch.autograd import Variable
import os
import kornia as K
import torch.nn.functional as F
from core.utils import frame_utils



def generate_refer_grid(imgs, num_imgs, device=None):

    if device == None:
        b, _, h, w = imgs.shape
        grid_x = torch.arange(w)
        grid_y = torch.arange(h)
        yy, xx = torch.meshgrid(grid_y, grid_x)
        refer_grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1).repeat(num_imgs, 1, 1, 1).float()

        refer_grid[..., 0] = refer_grid[..., 0] - (refer_grid.shape[2] // 2 - 0.5)
        refer_grid[..., 1] = refer_grid[..., 1] - (refer_grid.shape[1] // 2 - 0.5)
    else:
        b, _, h, w = imgs.shape
        grid_x = torch.arange(w, device=device)
        grid_y = torch.arange(h, device=device)
        yy, xx = torch.meshgrid(grid_y, grid_x)
        refer_grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1).repeat(num_imgs, 1, 1, 1).float()

        refer_grid[..., 0] = refer_grid[..., 0] - (refer_grid.shape[2] // 2 - 0.5)
        refer_grid[..., 1] = refer_grid[..., 1] - (refer_grid.shape[1] // 2 - 0.5)

    norm_grid_x = refer_grid[..., 0] / (refer_grid.shape[2] // 2)
    norm_grid_y = refer_grid[..., 1] / (refer_grid.shape[1] // 2)

    norm_grid = torch.stack([norm_grid_x, norm_grid_y], -1)

    return refer_grid, norm_grid

def generate_kernel_grid(size, num_imgs, device):

    grid_x = torch.arange(size)
    grid_y = torch.arange(size)
    yy, xx = torch.meshgrid(grid_y, grid_x)
    refer_grid = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1).repeat(num_imgs, 1, 1, 1).float()

    refer_grid[..., 0] = refer_grid[..., 0] - (refer_grid.shape[2] / 2 - 0.5)
    refer_grid[..., 1] = refer_grid[..., 1] - (refer_grid.shape[1] / 2 - 0.5)

    return refer_grid.to(device)


def save_datas(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./save_datas/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    # print('check is over')



def save_motion_datas(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./motion_datas/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')



def save_fore_datas(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./fore_saves/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')


def save_fore_modi_datas(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./fore_modi_saves/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')


def fore_saves_modi_two(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./fore_saves_modi_two/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')



def fore_saves_modi_three(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    save_folder = "./fore_saves_modi_three_newraft/" + "%06d" % save_ind
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')


def fore_saves_modi_four(ref_img, next_img, flows, outlier, param, save_ind, args, before=True):

    b, _, _, _ = ref_img.shape

    # save_folder = "./fore_saves_modi_four/" + "%06d" % save_ind
    save_origin_foler = "./" + args.save_dir
    if not os.path.isdir(save_origin_foler):
        os.mkdir(save_origin_foler)
    save_folder = save_origin_foler + "/" + "%06d" % save_ind

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu() * 64, dtype=np.int16)
    outlier = np.asarray((outlier * 255.0).detach().cpu(), dtype=np.uint8)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, outlier)
    np.save(flow_name, flows[0])

    for key in param.keys():
        save_name = save_folder + "/" + str(key)
        np.save(save_name, param[key])
    print('check is over')

def get_parameters(device):
    rand_motions = ((torch.rand((1, 2), dtype=torch.float32, device=device) - 0.5) * 2)
    affine_centers = ((torch.rand((1, 2), dtype=torch.float32, device=device) - 0.5) * 2)
    zoom_factors = torch.rand((1, 2), dtype=torch.float32, device=device) * 0.5 + 0.75
    rotations = ((torch.rand((1,), dtype=torch.float32, device=device) - 0.5) * 60)

    return rand_motions, affine_centers, zoom_factors, rotations


def get_back_parameters(device):
    rand_motions = ((torch.rand((1, 2), dtype=torch.float32, device=device) - 0.5) * 2).requires_grad_()
    affine_centers = ((torch.rand((1, 2), dtype=torch.float32, device=device, requires_grad=True) - 0.5) * 2).requires_grad_()
    zoom_factors = 2 ** (torch.rand((1,))*0.6 - 0.2)
    zoom_factors = torch.stack([zoom_factors, zoom_factors], -1)

    rotations = ((torch.rand((1,), dtype=torch.float32, device=device, requires_grad=True) - 0.5) * 2).requires_grad_()

    return rand_motions, affine_centers, zoom_factors, rotations


def augmentation(input1, input2, flow, outliers, batch_size):
    input1 = input1.repeat(batch_size, 1, 1, 1)
    input2 = input2.repeat(batch_size, 1, 1, 1)
    flow = flow.repeat(batch_size, 1, 1, 1)
    outliers = outliers.repeat(batch_size, 1, 1, 1)

    augmented_input1 = []
    augmented_input2 = []
    augmented_flow = []
    augmented_outliers = []

    for idx in range(batch_size):
        img1 = input1[idx].unsqueeze(0)
        img2 = input2[idx].unsqueeze(0)
        gt_flo = flow[idx].unsqueeze(0)
        outlier = outliers[idx].unsqueeze(0)

        b, c, ht, wd = img1.shape

        asymmetric_color_aug_prob = np.random.uniform(0, 1) < 0.2
        stretch_prob = np.random.uniform(0, 1) < 0.8
        spatial_aug_prob = np.random.uniform(0, 1) < 0.8
        h_flip_prob = np.random.uniform(0, 1) < 0.5
        max_stretch = 0.2

        if asymmetric_color_aug_prob:
            aug = K.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.125 / 3.14, p=1.0)
            img1 = aug(img1)
            img2 = aug(img2)
        else:
            aug = K.augmentation.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.125 / 3.14, p=1.0)
            img1 = aug(img1)
            img2 = aug(img2, params=aug._params)

        size_min_scale = np.maximum(
            (672 + 8) / float(ht),
            (1200 + 8) / float(wd))

        max_scale = 1.0
        min_scale = 0.1

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        scale_x = scale
        scale_y = scale

        if stretch_prob:
            scale_x *= 2 ** np.random.uniform(-max_stretch, max_stretch)
            scale_y *= 2 ** np.random.uniform(-max_stretch, max_stretch)

        scale_x = np.clip(scale_x, size_min_scale, None)
        scale_y = np.clip(scale_y, size_min_scale, None)

        if spatial_aug_prob:
            temp_input1 = F.interpolate(img1, scale_factor=[scale_y, scale_x], mode='bilinear')
            temp_input2 = F.interpolate(img2, scale_factor=[scale_y, scale_x], mode='bilinear')
            temp_flow = F.interpolate(gt_flo, scale_factor=[scale_y, scale_x], mode='bilinear')
            temp_outlier = F.interpolate(outlier, scale_factor=[scale_y, scale_x], mode='bilinear')
            temp_outlier = (temp_outlier > 0.95) * 1.0

            temp_flow_x = temp_flow[:, :1, ...] * scale_x
            temp_flow_y = temp_flow[:, 1:, ...] * scale_y
            temp_flow_complete = torch.cat([temp_flow_x, temp_flow_y], dim=1)

            img1 = temp_input1
            img2 = temp_input2
            gt_flo = temp_flow_complete
            outlier = temp_outlier

        if h_flip_prob:
            temp_input1 = K.geometry.hflip(img1)
            temp_input2 = K.geometry.hflip(img2)
            temp_flow2 = K.geometry.hflip(gt_flo)
            temp_outlier = K.geometry.hflip(outlier)

            temp_flow_x = temp_flow2[:, :1, ...] * -1
            temp_flow_y = temp_flow2[:, 1:, ...]
            temp_flow_complete = torch.cat([temp_flow_x, temp_flow_y], dim=1)

            img1 = temp_input1
            img2 = temp_input2
            gt_flo = temp_flow_complete
            outlier = temp_outlier

        bb, cc, hh, ww = img1.shape

        y0 = np.random.randint(0, hh - 672)
        x0 = np.random.randint(0, ww - 1200)

        output1 = img1[:, :, y0:y0+672, x0:x0+1200]
        output2 = img2[:, :, y0:y0+672, x0:x0+1200]
        output_flo = gt_flo[:, :, y0:y0+672, x0:x0+1200]
        out_outlier = outlier[:, :, y0:y0+672, x0:x0+1200]

        augmented_input1.append(output1)
        augmented_input2.append(output2)
        augmented_flow.append(output_flo)
        augmented_outliers.append(out_outlier)

    input1_outputs = torch.cat(augmented_input1, dim=0)
    input2_outputs = torch.cat(augmented_input2, dim=0)
    flow_outputs = torch.cat(augmented_flow, dim=0)
    outlier_outputs = torch.cat(augmented_outliers, dim=0)

    return input1_outputs, input2_outputs, flow_outputs, outlier_outputs


def generate_random_fog(back_img, device):

    _, _, h, w = back_img.shape
    a4 = torch.empty(h // 64, w // 64).normal_(mean=0, std=7).unsqueeze(0).unsqueeze(0).to(device)
    a3 = torch.empty(h // 32, w // 32).normal_(mean=0, std=5).unsqueeze(0).unsqueeze(0).to(device)
    a2 = torch.empty(h // 16, w // 32).normal_(mean=0, std=3).unsqueeze(0).unsqueeze(0).to(device)
    a1 = torch.empty(h // 8, w // 16).normal_(mean=0, std=2).unsqueeze(0).unsqueeze(0).to(device)
    a = torch.empty(h // 4, w // 4).normal_(mean=0, std=0.5).unsqueeze(0).unsqueeze(0).to(device)
    b = torch.empty(h // 2, w // 2).normal_(mean=0, std=0.25).unsqueeze(0).unsqueeze(0).to(device)

    Fog = F.interpolate(b, size=(h, w)) + F.interpolate(a, size=(h, w)) + F.interpolate(a2, size=(h, w),mode='bilinear') + \
          F.interpolate(a1, size=(h, w), mode='bilinear') + F.interpolate(a3, size=(h, w), mode='bilinear') \
          + F.interpolate(a4, size=(h, w),mode='bilinear')
    Fog = Fog / torch.max(Fog)

    random_fog_intensity = np.random.uniform(0, 1)
    Fog = Fog.clamp(0, 1) * random_fog_intensity

    return Fog


def save_subset(ref_img, next_img, outlier, flows, save_total_loss, save_ind, args):

    b, _, _, _ = ref_img.shape

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    save_origin_foler = os.path.join(args.save_dir, "best_tloss")
    if not os.path.isdir(save_origin_foler):
        os.mkdir(save_origin_foler)

    save_name = "%06d" % save_ind
    save_folder = os.path.join(save_origin_foler, save_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    outlier = outlier.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    outlier = np.asarray(outlier.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu(), dtype=np.float32)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow.flo"
    alpha_name = save_folder + "/total_loss"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, cv2.cvtColor(outlier[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    frame_utils.writeFlow(flow_name, flows[0])

    total_loss = np.array(save_total_loss.detach().cpu(), dtype=np.float32)
    f = open(alpha_name, 'w')
    f.write("{:.5f}".format(total_loss))
    f.close()


def save_wsl_subset(ref_img, next_img, outlier, flows, save_total_loss, save_ind, args):

    b, _, _, _ = ref_img.shape

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    save_origin_foler = os.path.join(args.save_dir, "best_wsl")
    if not os.path.isdir(save_origin_foler):
        os.mkdir(save_origin_foler)

    save_name = "%06d" % save_ind
    save_folder = os.path.join(save_origin_foler, save_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    outlier = outlier.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    outlier = np.asarray(outlier.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu(), dtype=np.float32)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow.flo"
    alpha_name = save_folder + "/total_loss"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, cv2.cvtColor(outlier[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    frame_utils.writeFlow(flow_name, flows[0])

    total_loss = np.array(save_total_loss.detach().cpu(), dtype=np.float32)
    f = open(alpha_name, 'w')
    f.write("{:.5f}".format(total_loss))
    f.close()


def save_tepe_subset(ref_img, next_img, outlier, flows, save_total_loss, save_ind, args):

    b, _, _, _ = ref_img.shape

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    save_origin_foler = os.path.join(args.save_dir, "best_tepe")
    if not os.path.isdir(save_origin_foler):
        os.mkdir(save_origin_foler)

    save_name = "%06d" % save_ind
    save_folder = os.path.join(save_origin_foler, save_name)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    ref_img = ref_img.permute(0, 2, 3, 1)
    next_img = next_img.permute(0, 2, 3, 1)
    outlier = outlier.permute(0, 2, 3, 1)
    flows = flows.permute(0, 2, 3, 1)

    ref_img = np.asarray(ref_img.detach().cpu(), dtype=np.uint8)
    next_img = np.asarray(next_img.detach().cpu(), dtype=np.uint8)
    outlier = np.asarray(outlier.detach().cpu(), dtype=np.uint8)
    flows = np.asarray(flows.detach().cpu(), dtype=np.float32)

    refer_name = save_folder + "/img_0.jpg"
    next_name = save_folder + "/img_1.jpg"
    outlier_name = save_folder + "/img_2.jpg"
    flow_name = save_folder + "/flow.flo"
    alpha_name = save_folder + "/tepe"

    cv2.imwrite(refer_name, cv2.cvtColor(ref_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(outlier_name, cv2.cvtColor(outlier[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(next_name, cv2.cvtColor(next_img[0], cv2.COLOR_RGB2BGR))
    frame_utils.writeFlow(flow_name, flows[0])

    total_loss = np.array(save_total_loss.detach().cpu(), dtype=np.float32)
    f = open(alpha_name, 'w')
    f.write("{:.5f}".format(total_loss))
    f.close()