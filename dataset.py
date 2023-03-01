# %%
import os
from collections.abc import Sequence
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, InterpolationMode
from torchvision.transforms.transforms import Pad, Resize


def image_fetch(train_folder:Path, split_rate:float=0.9):
    image_train = []
    label_train = []
    image_val = []
    label_val = []

    # get images and gt folders
    img_folder = os.path.join(train_folder, f"image")
    gt_folder = os.path.join(train_folder, f"GT")

    # load images from their folder
    img_list = glob(os.path.join(img_folder, "*.png"))
    gt_list = []
    
    # load corresponding gt
    for i in img_list[:]:
        img_name = os.path.basename(i)
        gt_path = os.path.join(gt_folder, img_name)
        if os.path.exists(gt_path):
            gt_list.append(gt_path)
        else:
            img_list.remove(i)
    print(f"total train images: {len(img_list)}")

    # split train/val
    split = round(len(img_list) * split_rate)
    image_train += img_list[:split]
    label_train += gt_list[:split]
    image_val += img_list[split:]
    label_val += gt_list[split:]

    return image_train, label_train, image_val, label_val

# %%
class ResizeSquarePad(nn.Module):
    def __init__(self, target_length:int, interpolation_strategy:InterpolationMode, pad_value=0):
        super(ResizeSquarePad, self).__init__()
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError(
                "Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        self.pad_value = pad_value

    def forward(self, img:Image.Image):
        w, h = img.size
        if w > h:
            target_size = (
                int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize(size=target_size, interpolation=self.interpolation_strategy)(img)

            total_pad = img.size[0] - img.size[1]
            half_pad = total_pad // 2
            padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad(padding=padding, fill=self.pad_value)(img)
        else:
            target_size = (self.target_length, int(
                np.round(self.target_length * (w / h))))
            img = Resize(size=target_size, interpolation=self.interpolation_strategy)(img)

            total_pad = img.size[1] - img.size[0]
            half_pad = total_pad // 2
            padding = (half_pad, 0, total_pad - half_pad, 0)
            return Pad(padding=padding, fill=self.pad_value)(img)

# %%
class SegDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.transform_img = Compose([
            ResizeSquarePad(target_length=512, interpolation_strategy=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.transform_mask = Compose([
            ResizeSquarePad(512, InterpolationMode.NEAREST)
        ])

        self.image_list = []

        for i in range(len(image_list)):
            self.image_list.append((image_list[i], label_list[i]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path, label_path = self.image_list[idx]
        img = Image.open(img_path)
        mask = Image.open(label_path)
        try:
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64) # mask transform does not contain to_tensor function
        except Exception as e:
            print(img_path)
            print(e)

        return img, mask

# %%
class PredSegDataset(Dataset):
    def __init__(self, img_list):
        super(PredSegDataset, self).__init__()

        self.img_list = img_list
        self.transforms = Compose([
            ResizeSquarePad(target_length=512, interpolation_strategy=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    
    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img = self.img_list[index]
        img = self.transforms(img)
        return img
    
# %%
if __name__ == "__main__":
    image_train, label_train, image_val, label_val = image_fetch(f"./PNG")
    ds = SegDataset(image_train, label_train)
    a, b = ds[0]
