
import torch

import torch.nn as nn

from PIL import Image

import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.metrics import f1_score,precision_score,recall_score,classification_report

import numpy as np
from torch.utils.data import DataLoader
import os

# train

def train(args,train_data,model,processor,device,checkpoint=None,sampler=None,shuffle=False):
    dataTrainLoader=DataLoader(train_data,batch_size=args.train_batch_size,shuffle=shuffle,sampler=sampler)

    if checkpoint is None:

        start_epoch=0
        model.to(device)
        model.train()

        if args.optimizer_name == 'adam':

            from torch.optim import AdamW

            optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon,weight_decay=args.weight_decay)

            print("Used Adamw optimizer")

        elif args.optimizer_name == 'sgd':

            from torch.optim import SGD

            optimizer = SGD(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)

            print("Used SGD optimizer")

    else:

        start_epoch=checkpoint['epoch']+1

        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)

        if args.optimizer_name == 'adam':

            from torch.optim import AdamW

            optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon,weight_decay=args.weight_decay)

            #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("Used Adamw optimizer")

        elif args.optimizer_name == 'sgd':

            from torch.optim import SGD

            optimizer = SGD(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)

            print("Used SGD optimizer")

    try:

        for i_epoch in range(start_epoch,start_epoch+args.num_train_epochs):
            
            

            sum_loss=0

            sum_step=0

            predict=None

            target=None

            for step,batch in enumerate(dataTrainLoader):

                image,caption,ocr,label=batch

                inputs=processor.forward(annotation=caption,ocr=ocr,image=image)

                label=torch.tensor(label).to(device)



                loss,score=model(inputs,label)



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                sum_loss+=loss.item()

                sum_step+=1

                outputs=torch.argmax(score,dim=-1)

                if predict is None:

                    predict=outputs

                    target=label

                else:

                    predict=torch.cat((predict,outputs),dim=0)

                    target=torch.cat((target,label),dim=0)

                if(step%100==0):

                    print("loss: ",sum_loss/sum_step)

                    print("f1_score: ",f1_score(target.cpu().numpy(),predict.cpu().numpy(),average="macro"))

        

            print("epoch: ",i_epoch,"loss: ",sum_loss/sum_step)

            print("f1_score: ",f1_score(target.cpu().numpy(),predict.cpu().numpy(),average="macro"))

            print("classification_scroce\n: ",classification_report(target.cpu().numpy(),predict.cpu().numpy()))

            path_to_save = os.path.join(args.output_dir, args.model)

            if not os.path.exists(path_to_save):
    
                os.makedirs(path_to_save,exist_ok=True)
    
            model_to_save = (model.module if hasattr(model, "module") else model)
    
            torch.save({
    
                        'epoch': i_epoch,
    
                        'model_state_dict': model_to_save.state_dict(),
    
                        'optimizer_state_dict': optimizer.state_dict(),  # lưu trạng thái optimizer
    
                        'loss': loss.item()  # lưu giá trị loss cuối cùng của epoch
    
                    }, os.path.join(path_to_save, 'model_epoch_{}.pt'.format(i_epoch)))
    
            print("Done save checkpoint at {} epoch".format(i_epoch))

    except Exception as e:

        print("Error during training:", e)

        path_to_save = os.path.join(args.output_dir, args.model)

        if not os.path.exists(path_to_save):

            os.makedirs(path_to_save, exist_ok=True)

        model_to_save = model.module if hasattr(model, "module") else model

        torch.save({

            'epoch': i_epoch,

            'model_state_dict': model_to_save.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'loss': loss.item()

        }, os.path.join(path_to_save, 'model_epoch_{}.pt'.format(i_epoch)))

        print("Done saving checkpoint at {} epoch".format(i_epoch))


# evaluate

def evaluate_score(args, model, device, data, processor, macro=False,pre = None, mode='test'):

  data_loader=DataLoader(data,batch_size=args.dev_batch_size,shuffle=False)

  n_correct,n_total=0,0

  t_target_all,t_output_all=None,None

  model.to(device)

  model.eval()

  sum_loss=0

  sum_step=0

  with torch.no_grad():

    for step,batch in enumerate(data_loader):

        image,caption,ocr,labels=batch

        inputs=processor.forward(annotation=caption,ocr=ocr,image=image)

        labels=torch.tensor(labels).to(device)



        t_targets=labels

        loss,score=model(inputs,labels)

        sum_loss+=loss.item()

        sum_step+=1

        #print(score.shape,labels.shape)



        outputs=torch.argmax(score,-1)
        n_correct+=(outputs==labels).sum().item()

        n_total+=len(labels)



        if t_target_all is None:

            t_target_all=t_targets

            t_output_all=outputs

        else:

            t_target_all=torch.cat((t_target_all,t_targets),dim=0)

            t_output_all=torch.cat((t_output_all,outputs),dim=0)

    if mode == 'test':

        print("test loss: ",sum_loss/sum_step)

    else:

        print("dev loss: ",sum_loss/sum_step)

    predict=t_output_all.cpu().numpy()

    label=t_target_all.cpu().numpy()

    if not macro:

        acc = n_correct / n_total

        f1 = f1_score(label,predict,average='micro')

        precision =  precision_score(label,predict,average='micro')

        recall = recall_score(label,predict,average='micro')

    else:

        acc = n_correct / n_total

        f1 = f1_score(label,predict,average='macro')

        precision =  precision_score(label,predict,average='macro')

        recall = recall_score(label,predict,average='macro')

    return acc, f1 ,precision,recall