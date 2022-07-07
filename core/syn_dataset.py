import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import kornia as K

import os
from glob import glob
import os.path as osp


class backDataset(data.Dataset):
    def __init__(self):
        self.image_list = []
        self.h_hlip = []
        self.gauss = []
        self.gen_index = 0
        self.h_size = 720 + 200 + 100 
        self.w_size = 1280 + 200 + 100
        self.crop_h_size = 720 + 200
        self.crop_w_size = 1280 + 200
    def __getitem__(self, index):

        img = Image.open(self.image_list[index])

        width, height = img.size

        if width < self.w_size or height < self.h_size:
            img = transforms.Resize(size=(self.h_size, self.w_size))(img)
            img = transforms.RandomCrop(size=(self.crop_h_size, self.crop_w_size))(img)
        else:
            img = transforms.RandomCrop(size=(self.crop_h_size, self.crop_w_size))(img)

        if np.random.uniform(0, 1) < 0.5:
            img = transforms.functional.hflip(img)

        img = transforms.ToTensor()(img)[:3, ...]

        if img.shape[0] != 3:
            img = img.repeat(3, 1, 1)

        img = torch.cat([img, torch.ones([1, img.shape[1], img.shape[2]])], dim=0)

        return img

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class COCO(backDataset):
    def __init__(self, root="/local_data/kwon/dupdate_images/COCO/train2017"):
        super(COCO, self).__init__()
        images = sorted(glob(osp.join(root, '*.jpg')))
        for i in range(len(images)):
            self.image_list += [images[i]]


class OpenImages(backDataset):
    def __init__(self, root="/local_data/kwon/dupdate_images/openimages"):
        super(OpenImages, self).__init__()
        images = sorted(glob(osp.join(root, '*.jpg')))
        for i in range(len(images)):
            self.image_list += [images[i]]


class DAVIS(backDataset):
    def __init__(self, root="/local_data/kwon/dupdate_images/DAVIS/JPEGImages/Full-Resolution"):
        super(DAVIS, self).__init__()

        for scene in os.listdir(root):
            image_list = sorted(glob(osp.join(root, scene, "*.jpg")))
            for i in range(len(image_list)):
                self.image_list += [image_list[i]]


class MonKaa(backDataset):
    def __init__(self, root="/local_data/kwon/dupdate_images/monkaa_finalpass"):
        super(MonKaa, self).__init__()

        for scene in os.listdir(root):
            image_list = sorted(glob(osp.join(root, scene, "left", "*.png")))
            for i in range(len(image_list)):
                self.image_list += [image_list[i]]


def back_dataloader():

    coco_data = COCO()
    print('Use %d COCO as background!' % len(coco_data))
    openimages_data = OpenImages()
    print('Use %d OpenImages as background!' % len(openimages_data))
    davis_data = DAVIS()
    print('Use %d DAVIS as background!' % len(davis_data))
    monkaa_data = MonKaa()
    print('Use %d MonKaa as background!' % len(monkaa_data))
    total_dataset = coco_data + openimages_data + davis_data + monkaa_data
    back_loader = data.DataLoader(total_dataset, num_workers=0, batch_size=1+8+4, pin_memory=False, shuffle=True, drop_last=True)

    return back_loader


class foreData(data.Dataset):
    def __init__(self,
                 root="/local_data/kwon/dupdate_images/voc_non_processing/"):
        super(foreData, self).__init__()

        self.fore_size = 360
        self.imgs = []
        self.h_hlip = []
        self.h_size = []
        self.w_size = []
        self.gen_index = 0

        file_list = os.listdir(root)
        imgs = [file for file in file_list if file.endswith(".png")]

        imgs = sorted(imgs)
        for i in range(len(imgs)):
            self.imgs.append(root + imgs[i])

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])

        if np.random.uniform(0, 1) < 0.5:
            img = transforms.functional.hflip(img)

        img = transforms.ToTensor()(img)[-1:, ...].unsqueeze(0)

        aug = K.augmentation.RandomAffine(degrees=180., scale=(0.9, 1.1), shear=30, p=1)
        img = aug(img).squeeze(0)

        _, h, w = img.shape
        max_len = max(h, w)
        s_limit = min(360 / max_len, 2)
        s_min = min(0.8, 0.7 * s_limit)

        s_factor = np.random.uniform(low=s_min, high=s_limit)
        h_size = int(img.shape[1]*s_factor)
        w_size = int(img.shape[2]*s_factor)

        img = F.interpolate(img.clone().unsqueeze(0), size=[h_size, w_size], mode='bilinear', align_corners=False).clamp(0, 1)

        _, _, h, w = img.shape

        pad1 = (self.fore_size - w) // 2
        pad2 = (self.fore_size - w) // 2 + (self.fore_size - w) % 2
        pad3 = (self.fore_size - h) // 2
        pad4 = (self.fore_size - h) // 2 + (self.fore_size - h) % 2

        img = F.pad(img.clone(), pad=[pad1, pad2, pad3, pad4], mode='constant', value=0).squeeze(0)

        return img

    def __len__(self):
        return len(self.imgs)


def fore_dataloader():
    fore_dataset = foreData()
    print('Use %d VOC as foreground!' % len(fore_dataset))
    fore_loader = data.DataLoader(fore_dataset, batch_size=8+4, num_workers=0, pin_memory=False, shuffle=True, drop_last=True)

    return fore_loader

