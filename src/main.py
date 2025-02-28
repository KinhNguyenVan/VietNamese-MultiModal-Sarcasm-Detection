from model import MultimodalModel
from train import train
from data_set import CaptionImageDataset
from processor import processor
import torch
import argparse
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

from torchvision import transforms

from PIL import Image

import os

import json

def set_args():

    """

    Sets the script's arguments using argparse.



    Instead of parsing command-line arguments,

    we'll manually create an argument namespace with

    the default values. This simulates what would happen

    if the script was run from the command line with no

    additional arguments.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')

    parser.add_argument('--model', default='multimodal', type=str, help='model name')

    parser.add_argument('--output_dir', default='/checkpoint/', type=str, help='output directory')

    parser.add_argument('--train_batch_size', default=16, type=int, help='batch size in train phase')

    parser.add_argument('--dev_batch_size', default=16, type=int, help='batch size in dev phase')

    parser.add_argument('--text_size', default=768, type=int, help='text hidden size')

    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--optimizer_name", type=str, default='adam',help="use which optimizer to train the model.")

    parser.add_argument('--layers1', default=1, type=int, help='number of transform layers ocr-image')

    parser.add_argument('--layers2', default=1, type=int, help='number of transform layers text-image-annotation')

    parser.add_argument('--layers3', default=1, type=int, help='number of transform layers ocr-annotation')

    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')

    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate')

    parser.add_argument('--is_train',default=False,type=bool,help='freeze the backbone during training')

    parser.add_argument('--num_classes',default=4,type=int,help='number of classes')

    parser.add_argument('--weight_decay',default=0.1,type=float,help='regularization')



    args = parser.parse_args(args=[])

    return args

if __name__=='__main__':

    transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()

    ])

    label2Idx = {'text-sarcasm': 0, 'image-sarcasm': 1, 'multi-sarcasm': 2, 'not-sarcasm': 3}

    # Khởi tạo tập train và test

    trainset = CaptionImageDataset(

        json_file='.../data/train/vimmsd-train.json',

        json_ocr=".../data/train/output_data2.json",

        img_dir='.../data/train/train-images/train-images',

        transform=transform,

        num_classes=4,

        label2Idx=label2Idx,

        train=True

    )


    with open(".../data/train/vimmsd-train.json","r") as f:

        data=json.load(f)

    labels=[label2Idx[data[str(idx)]['label']] for idx in range(len(trainset))]

    labels = torch.tensor(labels, dtype=torch.long)

    class_sample_counts = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])

    weights = 1. / class_sample_counts.float()  

    sample_weights = torch.tensor([weights[t] for t in labels]) 


    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessor=processor(device=device)

    args=set_args()

    args.output_dir="..."

    model=MultimodalModel(args)

    train(args,trainset,model,preprocessor,device,sampler=sampler)
