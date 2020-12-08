# %% import dependencies
import os
os.chdir('/home/zhaozixiao/projects/UNet')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch import nn
import torchvision
from torchvision.transforms.transforms import Pad, Resize
from torch.utils.data import DataLoader
from collections.abc import Sequence
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

from unet_resnet import ResNetUNet
from dataset import *

# %% load model
test_model = ResNetUNet(21)
test_model.cuda()
test_model.load_state_dict(torch.load("/data1/zhaozixiao/projects/UNet/model_resunet/model_512_resunet_49.pth"))
print("model loaded!")

# %% define transform function
class ResizeSquarePad(Resize, Pad):
    def __init__(self, target_length, interpolation_strategy):
        if not isinstance(target_length, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(target_length)))
        if isinstance(target_length, Sequence) and len(target_length) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        Resize.__init__(self, size=(512, 512), interpolation=self.interpolation_strategy)
        Pad.__init__(self, padding=(0,0,0,0), fill=255, padding_mode="constant")


    def __call__(self, img):
        w, h = img.size
        if w > h:
            # if image width > height, resize while keeping w-h ratio and pad in vertical direction
            self.size = (int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize.__call__(self, img)

            total_pad = self.size[1] - self.size[0]
            half_pad = total_pad // 2
            self.padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad.__call__(self, img)
        else:
            # else resize while keeping w-h ratio and pad in horizontal direction
            self.size = (self.target_length, int(np.round(self.target_length * (w / h))))
            img = Resize.__call__(self, img)

            total_pad = self.size[0] - self.size[1]
            half_pad = total_pad // 2
            self.padding = (half_pad, 0, total_pad - half_pad, 0)
            return Pad.__call__(self, img)

transform_img = torchvision.transforms.Compose([
    ResizeSquarePad(512, Image.BILINEAR),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform_mask = torchvision.transforms.Compose([
    ResizeSquarePad(512, Image.NEAREST) # attention here, resize mask need to use NEAREST to keep the label dtype as int
])
# %% test one image
ori_image = Image.open("/data1/zhaozixiao/projects/UNet/voc.devkit/voc2012/val/image/2008_002993.jpg")
ori_mask = Image.open("/data1/zhaozixiao/projects/UNet/voc.devkit/voc2012/val/mask/2008_002993.png")

image = transform_img(ori_image)
mask = transform_mask(ori_mask)
mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

image = image.cuda()
predict = test_model(image.unsqueeze(0))

predict = predict.squeeze(0)
predict = nn.Softmax(dim=0)(predict)
predict = torch.argmax(predict, dim=0)

acc = {}
pure_mask = mask.masked_select(mask.ne(255))
pure_predict = predict.masked_select(mask.ne(255))
acc['overall'] = pure_mask.eq(pure_predict.cpu()).sum().item()/len(pure_mask)

for class_value in pure_mask.unique():
    valued_mask = pure_mask.masked_select(pure_mask.eq(class_value))
    valued_predict = pure_predict.masked_select(pure_mask.eq(class_value))
    acc[class_value.item()] = valued_mask.eq(valued_predict.cpu()).sum().item()/len(valued_mask)

w, h = ori_image.size
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

for key in acc:
    print(str(key) + " accuracy: " + str(acc[key]))
plt.imshow(out)
plt.imshow

# %% test on all val images
val_image_dir = "./voc.devkit/voc2012/val/"
X_val, y_val = valImageFetch(val_image_dir)
val_data = SaltDataset(X_val, y_val, 'val', transform_img, transform_mask)

val_loader = DataLoader(val_data,
                    shuffle=False,
                    batch_size=4)


device = torch.device('cuda')

temp_acc = {}
for i in range(21):
    temp_acc[i] = [0, 0.0]
temp_acc['overall'] = [0, 0.0]


test_model.eval()

with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.long().to(device)

        outputs = test_model(inputs)

        predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
        pure_mask = masks.masked_select(masks.ne(255))
        pure_predict = predict.masked_select(masks.ne(255))
        temp_acc['overall'][1] += pure_mask.eq(pure_predict).sum().item()/len(pure_mask)
        temp_acc['overall'][0] += 1

        for class_value in pure_mask.unique():
            valued_mask = pure_mask.masked_select(pure_mask.eq(class_value))
            valued_predict = pure_predict.masked_select(pure_mask.eq(class_value))
            temp_acc[class_value.item()][1] += valued_mask.eq(valued_predict).sum().item()/len(valued_mask)
            temp_acc[class_value.item()][0] += 1

for key in temp_acc.keys():
    print(f"class {key} accuracy: {temp_acc[key][1] / temp_acc[key][0]}")
