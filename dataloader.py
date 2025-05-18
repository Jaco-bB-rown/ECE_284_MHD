import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
import torchvision.transforms.v2 as v2

class ISICMaskImageDataset(Dataset):
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
    
class ISICClassImageDataset(Dataset):
    def __init__(self, dir, name, data_aug_type, transform=None, target_transform=None, size = (256,256)):
        self.img_labels = pd.read_csv(os.path.join(dir, name+"_GroundTruth.csv"))
        #bounding box labels
        self.bb_labels = pd.read_csv(os.path.join(dir, "Pred_bb\\"+name+"_bb_"+data_aug_type+".csv"))
        self.img_dir = os.path.join(dir,"Images\\",name)
        self.transform = transform
        self.target_transform = target_transform
        self.size = size

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = decode_image(img_path)
        x1,y1,x2,y2 = (self.bb_labels.iloc[idx,1],self.bb_labels.iloc[idx,2],self.bb_labels.iloc[idx,3],self.bb_labels.iloc[idx,4])
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
            image = v2.functional.resized_crop(image,top=int(y1),left=int(x1),height=int(y2-y1),width=int(x2-x1),size=self.size)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label