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



def predict(args, model, device, data, processor):

  data_loader=DataLoader(data,batch_size=args.dev_batch_size,shuffle=False)

  t_output_all=None

  model.to(device)

  model.eval()

  with torch.no_grad():

    for step,batch in enumerate(data_loader):

      image,caption,ocr,_=batch

      inputs=processor.forward(annotation=caption,ocr=ocr,image=image)

      score=model(inputs)

      outputs=torch.argmax(score,-1)

      if t_output_all is None:

        t_output_all=outputs

      else:

        t_output_all=torch.cat((t_output_all,outputs),dim=0)

    return t_output_all
  
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
  testset = CaptionImageDataset(

        json_file='D:/MMSD/data/test/vimmsd-private-test.json',

        json_ocr="D:/MMSD\data/test/private_ocr.json",
        img_dir='D:/MMSD/data/test/private-test-images',

        transform=transform,

        num_classes=4,

        label2Idx=label2Idx,

        train=False

    )
  args=set_args()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=MultimodalModel(args)

model.to(device)

checkpoint=torch.load("D:/MMSD/checkpoint/model_epoch_15.pt",weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])
preprocessor=processor(device=device)
y_pred=predict(args, model, device, testset, preprocessor)
