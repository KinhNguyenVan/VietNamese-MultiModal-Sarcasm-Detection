from collections import defaultdict

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

from torchvision import transforms

from PIL import Image

import os

import json

import torch



class CaptionImageDataset(Dataset):

    def __init__(self, json_file, json_ocr, img_dir, transform=None, num_classes=4, label2Idx=None, train=True):

        with open(json_file, 'r',encoding='utf-8') as f:

            self.data = json.load(f)

        with open(json_ocr, 'r',encoding='utf-8') as f:

            self.data_ocr = json.load(f)

        self.img_dir = img_dir

        self.transform = transform

        self.num_classes = num_classes

        self.label2Idx = label2Idx

        self.train = train



    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        item = self.data[str(idx)]

        img_name = item['image']

        caption = item['caption']

        label = item.get('label', None)

        

        if img_name == self.data_ocr[str(idx)]['image']:

            ocr = self.data_ocr[str(idx)]['ocr']

        else:

            raise ValueError("image name không map với file ocr!")

        if ocr == "":

            ocr = "None"

        

        label_int = self.label2Idx[label] if self.train else -1



        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")



        if self.transform:

            image = self.transform(image)



        return image, caption, ocr, label_int




