from transformers import DeiTFeatureExtractor, DeiTConfig, DeiTModel, AutoModel, AutoTokenizer,DeiTImageProcessor

import torch

import torch.nn as nn

from PIL import Image

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


class processor(nn.Module):

  def __init__ (self,device,llm_name="uitnlp/visobert"):

    self.tokenizer=AutoTokenizer.from_pretrained(llm_name)

    self.preprocessor=DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

    self.device=device

  def tokenize(self,text):

    tokens=self.tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=512)

    return tokens.to(self.device)

  def image_econde(self,image):

    inputs=self.preprocessor(images=image,return_tensors="pt")

    return inputs.to(self.device)

  def forward(self,annotation,ocr,image):

    output={}

    output['annotation']=self.tokenize(annotation)

    output['ocr']=self.tokenize(ocr)

    output['image']=self.image_econde(image)

    return output