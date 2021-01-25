# UNet_multiclass_segmentation_pytorch
An simple implementaion of PyTorch UNet segmentation model on VOC2012 dataset without any complicated structure, can be used directly.

## Requirements
torch == 1.6.0
torchvision == 0.7.0

## File Format
The training and validation set should be split into two folders separetely, 

```
 + datasets
   + train 
    + images
     - 0001.jpg
     - 0002.jpg
     ...
    + masks
     - 0001.png
     - 0002.png
     ...
   + val 
    + images
     - 0003.jpg
     - 0004.jpg
     ...
    + masks
     - 0003.png
     - 0004.png
     ...
 - train.py
 - unet.py
 - test.py
```

## Usage
Modify the `num_classes` in `train.py` line 43, modify the train/val folders path in `train.py` line 98.

Then run
```
python train.py
```
