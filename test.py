# %% import dependencies
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
from torch import nn
import torchvision
from torchvision.transforms.transforms import Pad, Resize
from torch.utils.data import DataLoader
from collections.abc import Sequence
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

from dataset import PredSegDataset, image_fetch
from unet import UNet

# %%
palette = []
for i in range(256):
    palette.extend((i, i, i))
palette[:3*21] = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128]
                        ], dtype='uint8').flatten()

def put_palette(img):
    img = img.astype(np.uint8)
    res = Image.fromarray(img)
    res.putpalette(palette)
    return res

# %% load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
state_dict = torch.load("./unet_512_32.pth")

model = UNet(input_channels=3, num_classes=2)
model.load_state_dict(state_dict)
if device == torch.device("cuda"):
    model = nn.DataParallel(model)
    batch_size = torch.cuda.device_count() * 4
else:
    batch_size = 4
model.to(device)
model.eval()
print("model loaded!")

# %% test one image
ori_image = Image.open("/home/zhaozixiao/projects/MLFlow/PNG/image/7.png")
img_size=[ori_image.size]

infer_data = PredSegDataset([ori_image])
infer_loader = DataLoader(infer_data,
            shuffle=False, 
            batch_size=batch_size) 

res_tensor = list()
with torch.no_grad():
    for inputs in infer_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        outputs = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
        res_tensor.extend([*outputs])

res = list()
for i in range(len(res_tensor)):
    predict = res_tensor[i]

    # transform result image into original size
    w, h = img_size[i]
    if w > h:
        re_h = int(np.round(512 * (h / w)))
        total_pad = 512 - re_h
        half_pad = total_pad // 2
        out = predict[half_pad : half_pad + re_h, :]
    else:
        re_w = int(np.round(512 * (w / h)))
        total_pad = 512 - re_w
        half_pad = total_pad // 2
        out = predict[:, half_pad : half_pad + re_w]

    out = cv2.resize(out.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
    res.append(put_palette(out))

for idx, i in enumerate(res):
    i.save(f"./{idx}_pred.png")