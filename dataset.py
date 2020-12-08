from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from PIL import Image


def trainImageFetch(train_folder):
    image_train = []
    mask_train = []

    # load images and masks from their folders
    images_folder = os.path.join(train_folder, "image")
    masks_folder = os.path.join(train_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load train images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_train.append(image)

        mask = Image.open(mask_path)
        mask_train.append(mask)

    return image_train, mask_train


def valImageFetch(val_folder):
    image_val = []
    mask_val = []

    images_folder = os.path.join(val_folder, "image")
    masks_folder = os.path.join(val_folder, "mask")

    image_list = os.listdir(images_folder)
    for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc="load validation images......"):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name.split('.')[0] + ".png")

        image = Image.open(image_path)
        image_val.append(image)

        mask = Image.open(mask_path)
        mask_val.append(mask)

    return image_val, mask_val


class SegDataset(Dataset):
    def __init__(self, image_list, mask_list, mode, transform_img, transform_mask):
        self.mode = mode
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.imagelist = image_list
        self.masklist = mask_list


    def __len__(self):
        return len(self.imagelist)


    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            # label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)
            # print(f'before transform mask max: {np.array(mask).max()}')
            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
            # print(f'after transform mask max: {mask.max()}')

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            image = self.transform_img(image)

            mask = self.transform_mask(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

            # image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            return image, mask
