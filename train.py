# %% import dependencies
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from dataset import SegDataset, image_fetch
from unet import UNet


# %% load model
model = UNet(input_channels=3, num_classes=2)
if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_size = 7
        gpu_num = torch.cuda.device_count()
        batch_size = batch_size * gpu_num
        model = nn.DataParallel(model)
else:
    device = torch.device('cpu')
    batch_size = 16
model.to(device)
print(f"Training UNet model on device: {device}, batch_size (total): {batch_size}")

criterion = nn.CrossEntropyLoss(ignore_index=255)

optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', patience=5, threshold=1e-3)


# %% dataloader
train_image_dir = "./PNG"

image_train, label_train, image_val, label_val = image_fetch(f"./PNG")
train_data = SegDataset(image_train, label_train)
val_data = SegDataset(image_val, label_val)

train_loader = DataLoader(train_data,
                    shuffle=RandomSampler(train_data),
                    batch_size=batch_size)

val_loader = DataLoader(val_data,
                    shuffle=False,
                    batch_size=batch_size)


# %%
def train_one_epoch(train_loader:DataLoader, model:nn.Module, device:torch.device, optimizer:torch.optim.Optimizer, criterion:nn.Module):
    running_loss = 0.0

    model.train()
    for inputs, masks in tqdm(train_loader):
        inputs, masks= inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        
        logit = model(inputs)
        loss = criterion(logit.squeeze(1), masks.squeeze(1))
        loss.backward()
        optimizer.step()
        # print(loss.item())
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def valid(test_loader:DataLoader, model:nn.Module, device:torch.device, criterion:nn.Module):
        running_loss = 0.0
        acc = 0.0

        model.eval()

        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(device), masks.to(device)

                outputs = model(inputs)

                predict = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
                pure_mask = masks.masked_select(masks.ne(255))
                pure_predict = predict.masked_select(masks.ne(255))
                acc += pure_mask.cpu().eq(pure_predict.cpu()).sum().item()/len(pure_mask)
                
                loss = criterion(outputs.squeeze(1), masks.squeeze(1))           
                running_loss += loss.item()

        epoch_loss = running_loss / len(test_loader)
        accuracy = acc / len(test_loader)
        return epoch_loss, accuracy


# %%
best_acc = 0
writer = SummaryWriter("./log/UNet")

for epoch_ in range(50):
    train_loss = train_one_epoch(
            train_loader=train_loader, 
            model=model, 
            device=device, 
            optimizer=optimizer_ft, 
            criterion=criterion)
    val_loss, accuracy = valid(
            test_loader=val_loader, 
            model=model,
            device=device,
            criterion=criterion)
    exp_lr_scheduler.step(val_loss)

    writer.add_scalar('loss/train', train_loss, epoch_)
    writer.add_scalar('loss/valid', val_loss, epoch_)
    writer.add_scalar('accuracy', accuracy, epoch_)

    print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))
    if accuracy > best_acc:
        best_acc = accuracy
        best_epoch = epoch_
        best_model = model.module.state_dict()

torch.save(best_model, f"./unet_512_{best_epoch}.pth")
writer.close()
