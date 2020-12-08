# %% import dependencies
import os
os.chdir('/home/zhaozixiao/projects/UNet')
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

from unet import UNet
from dataset import *
from unet_resnet import ResNetUNet
from fcn import FCN

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.sampler import RandomSampler
from torch import nn
import torchvision
from torchvision.transforms.transforms import Pad, Resize
from collections.abc import Sequence
from tensorboardX import SummaryWriter

# %% hyper parameters
n_fold = 5

# batch_size = 32 # for FCN
# batch_size = 8 # for res_unet
batch_size = 4
epoch = 50
snapshot = 5

max_lr = 0.012
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4

device = torch.device('cuda')

weight_name = 'model_512_unet'


# %% import model
# salt = ResNetUNet(21)
# salt = FCN(21)
salt = UNet(21)
salt.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)

scheduler_step = epoch // snapshot

optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, salt.parameters()), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

# %% pre-process
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
            self.size = (int(np.round(self.target_length * (h / w))), self.target_length)
            img = Resize.__call__(self, img)

            total_pad = self.size[1] - self.size[0]
            half_pad = total_pad // 2
            self.padding = (0, half_pad, 0, total_pad - half_pad)
            return Pad.__call__(self, img)
        else:
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
    ResizeSquarePad(512, Image.NEAREST)
])

# %%
# Load data
train_image_dir = "./voc.devkit/voc2012/train/"
val_image_dir = "./voc.devkit/voc2012/val/"

X_train, y_train = trainImageFetch(train_image_dir)
X_val, y_val = valImageFetch(val_image_dir)

# %%
train_data = SaltDataset(X_train, y_train, 'train', transform_img, transform_mask)
val_data = SaltDataset(X_val, y_val, 'val', transform_img, transform_mask)

# %%
train_loader = DataLoader(train_data,
                    shuffle=RandomSampler(train_data),
                    batch_size=batch_size)

val_loader = DataLoader(val_data,
                    shuffle=False,
                    batch_size=batch_size)


# %%
def train(train_loader, model):
    running_loss = 0.0
    data_size = len(train_data)

    model.train()

    for inputs, masks in tqdm(train_loader):
        inputs, masks= inputs.to(device), masks.long().to(device)
        optimizer_ft.zero_grad()

        logit = model(inputs)

        loss = criterion(logit, masks.squeeze(1))
        loss.backward()
        optimizer_ft.step()
        # print(loss.item())
        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / data_size
    return epoch_loss


def test(test_loader, model):
    running_loss = 0.0
    acc = 0.0
    data_size = len(test_loader)

    model.eval()

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.long().to(device)

            outputs = model(inputs)

            predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
            pure_mask = masks.masked_select(masks.ne(255))
            pure_predict = predict.masked_select(masks.ne(255))
            acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask)
            
            loss = criterion(outputs.squeeze(1), masks.squeeze(1))           
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / data_size
    accuracy = acc / data_size
    return epoch_loss, accuracy #precision


# %%
num_snapshot = 0
best_acc = 0
writer = SummaryWriter("./log/UNet")


for epoch_ in range(epoch):
    train_loss = train(train_loader, salt)
    val_loss, accuracy = test(val_loader, salt)
    exp_lr_scheduler.step()

    writer.add_scalar('loss/train', train_loss, epoch_)
    writer.add_scalar('loss/valid', val_loss, epoch_)
    writer.add_scalar('accuracy', accuracy, epoch_)
    # writer.add_scalars('Val_loss', {'val_loss': val_loss}, n_iter)

    if accuracy > best_acc:
      best_acc = accuracy
      best_param = salt.state_dict()

    print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))
    torch.save(salt.state_dict(), weight_name + '_%d.pth' % epoch_)
writer.close()
