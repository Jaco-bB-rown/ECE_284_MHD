import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
import torchvision.transforms.v2 as v2
import torch
from transforms import dataTransforms

#Dataset for training and testing the mask model
class ISICMaskImageDataset(Dataset):
    def __init__(self, dir, name, data_aug_type="1", size = (256,256)):
        self.img_labels = pd.read_csv(os.path.join(dir,os.path.join("Ground_Truths", name+"_GroundTruth_"+data_aug_type+".csv")))
        self.img_dir = os.path.join(os.path.join(dir,"Images"),name)
        self.mask_dir = os.path.join(os.path.join(dir,"Masks"),name)
        self.transform = dataTransforms("1",size=size,mask=True)
        self.data_transform = dataTransforms(data_aug_type,size=size,mask=True)
        self.data_aug_type = data_aug_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = decode_image(img_path)
        mask_path = os.path.join(self.mask_dir, self.img_labels.iloc[idx, 0]+"_segmentation.png")
        mask = decode_image(mask_path)
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image)   
        mask  = self.transform(mask)
        if self.data_aug_type != "1" and self.img_labels.iloc[idx, 3] == 1:#only modify specific malignant images
            image = self.data_transform(image)
            mask = self.data_transform(mask)
        label = self.target_transform_rcnn(label)
        boxes = masks_to_boxes(mask)
        return image, {'masks': mask,'boxes': boxes, 'labels': label}
    
    #function to transform our labels into the neccessary types i.e. one-hot
    def target_transform_rcnn(self ,y):
        return torch.zeros(2, dtype=torch.int64).scatter_(dim=0, index=torch.tensor(y,dtype=torch.int64), value=1)
    
#Reduced Mask dataset for genreating bounding boxes (allows us to ignore correct labels)
class ISICMaskImageDataset_for_generation(Dataset):
    def __init__(self, dir, name, data_aug_type="1", size = (256,256)):
        self.img_labels = pd.read_csv(os.path.join(dir,os.path.join("Ground_Truths", name+"_GroundTruth_"+data_aug_type+".csv")))
        self.img_dir = os.path.join(os.path.join(dir,"Images"),name)
        self.transform = dataTransforms("1",size=size,mask=True)
        self.data_aug_type = data_aug_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = decode_image(img_path)
        image = self.transform(image)   
        if self.data_aug_type != "1" and self.img_labels.iloc[idx, 3] == 1:#only modify specific malignant images
            image = self.data_transform(image)
        return image
    
#Dataset for Training and testing the classification model
class ISICClassImageDataset(Dataset):
    def __init__(self, dir, name, data_aug_type = "1", size = (224,224),bb_data_type="1"):
        self.img_labels = pd.read_csv(os.path.join(dir,os.path.join("Ground_Truths", name+"_GroundTruth_"+data_aug_type+".csv")))
        #bounding box labels
        self.bb_labels = pd.read_csv(os.path.join(dir,os.path.join( "Pred_bb", name+"_bb_"+bb_data_type+".csv")))
        self.img_dir = os.path.join(dir,os.path.join("Images",name))
        self.transform = dataTransforms("1", size=size, mask=False)
        self.data_transform = dataTransforms(data_aug_type, size=size, mask=False)
        self.size = size
        self.data_aug_type = data_aug_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+".jpg")
        image = decode_image(img_path)
        x1,y1,x2,y2 = (self.bb_labels.iloc[idx,1],self.bb_labels.iloc[idx,2],self.bb_labels.iloc[idx,3],self.bb_labels.iloc[idx,4])
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image)
        if self.data_aug_type != "1" and self.img_labels.iloc[idx, 3] == 1:#only modify specific malignant images
            image = self.data_transform(image)
        image = v2.functional.resized_crop(image,top=int(y1),left=int(x1),height=int(y2-y1),width=int(x2-x1),size=self.size)
        label = self.target_transform_resnet(label)
        return image, label
    
    #function to transform our labels into the neccessary types i.e. one-hot
    def target_transform_resnet(self,y):
        return torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y,dtype=torch.int64), value=1)