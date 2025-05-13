import os
import pandas as pd
from torchvision.io import read_image, decode_image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes

class ISICImageDataset(Dataset):
    def __init__(self, dir, name, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(dir, name+"_GroundTruth.csv"))
        self.img_dir = os.path.join(dir,"Images\\",name)
        self.mask_dir = os.path.join(dir,"Masks\\",name)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = decode_image(img_path)
        mask_path = os.path.join(self.mask_dir, self.img_labels.iloc[idx, 0]+"_segmentation.png")
        mask = decode_image(mask_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)
        if self.target_transform:
            label = self.target_transform(label)
        boxes = masks_to_boxes(mask)
        '''target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["masks"] = mask'''
        return image, {'masks': mask,'boxes': boxes, 'labels': label}
    #label, mask, boxes,