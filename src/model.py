from transformers import DeiTFeatureExtractor, DeiTConfig, DeiTModel, AutoModel, AutoTokenizer,DeiTImageProcessor

import torch

import torch.nn as nn

import copy

import torch.nn.functional as F

from torch.autograd import Variable
from peft import LoraConfig, get_peft_model

class MultimodalEncoder(nn.Module):

    def __init__(self, text_model, layer_number):

        super(MultimodalEncoder, self).__init__()

        layer_encoders = text_model.encoder

        self.layer = nn.ModuleList([copy.deepcopy(layer_encoders.layer[i]) for i in range(layer_number)])

        for layer in self.layer:  

            for param in layer.parameters():  

                param.requires_grad = True

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):

        all_encoder_layers = []

        all_encoder_attentions = []

        for layer_module in self.layer:

            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)

            all_encoder_attentions.append(attention)

            if output_all_encoded_layers:

                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:

            all_encoder_layers.append(hidden_states)

        return all_encoder_layers, all_encoder_attentions

class FeatureProjection(nn.Module):

  def __init__(self,hidden_size):

    super(FeatureProjection,self).__init__()

    self.linear=nn.Sequential(

        nn.Linear(hidden_size,hidden_size,bias=True),

        nn.LeakyReLU(),

    )

    self.norm=nn.BatchNorm1d(hidden_size)

  def forward(self,x):

    x=self.linear(x)

    batch_size, num_patches, hidden_size = x.shape

    x=x.view(batch_size*num_patches,hidden_size)

    x=self.norm(x)

    x=x.view(batch_size,num_patches,hidden_size)

    return x


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()

        self.gamma = gamma

        self.size_average = size_average

        if alpha is not None:

            if isinstance(alpha, (float, int)):

                # Trường hợp phân loại nhị phân

                self.alpha = torch.tensor([alpha, 1 - alpha])

            elif isinstance(alpha, list):

                # Trường hợp đa lớp, mỗi lớp có trọng số riêng

                self.alpha = torch.tensor(alpha)

        else:

            self.alpha = None



    def forward(self, input, target):

        # Chuyển đổi đầu vào nếu là tensor 4 chiều

        if input.dim() > 2:

            input = input.view(input.size(0), input.size(1), -1)  # N, C, H, W => N, C, H*W

            input = input.transpose(1, 2)  # N, C, H*W => N, H*W, C

            input = input.contiguous().view(-1, input.size(2))  # N, H*W, C => N*H*W, C

        target = target.view(-1, 1)



        # Tính log của softmax và chọn giá trị tương ứng với nhãn target

        logpt = F.log_softmax(input, dim=-1)

        logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)

        pt = logpt.exp()



        # Xử lý alpha cho trường hợp đa lớp

        if self.alpha is not None:

            if self.alpha.device != input.device:

                self.alpha = self.alpha.to(input.device)

            at = self.alpha.gather(0, target.view(-1))

            logpt = logpt * at



        # Tính loss focal

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:

            return loss.mean()

        else:

            return loss.sum()
        



class MultimodalModel(nn.Module):

  def __init__(self,args):

    super(MultimodalModel,self).__init__()

    self.text_model=AutoModel.from_pretrained("uitnlp/visobert",attn_implementation="eager")

    self.image_model=AutoModel.from_pretrained("facebook/dinov2-base")
    if args.lora is not True:
        for param in self.text_model.parameters():
            param.requires_grad=False
        for param in self.image_model.parameters():
            param.requires_grad=False
        for param in [self.text_model.pooler.dense.weight,
                      self.text_model.pooler.dense.bias]:
            param.requires_grad = True
        for param in [self.image_model.pooler.dense.weight,
                      self.image_model.pooler.dense.bias]:
            param.requires_grad = True
    else:
        lora_config_text = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value", "key","dense"],
                bias="none"
            )
        lora_config_image = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.1,
                target_modules=["query", "value", "key","dense"],
                bias="none"
            )
        self.text_model = get_peft_model(self.text_model, lora_config_text)
        self.image_model = get_peft_model(self.image_model,lora_config_image)

    self.transformer_I2A=MultimodalEncoder(self.text_model,args.layers1)

    self.transformer_I2C=MultimodalEncoder(self.text_model,args.layers2)

    self.transformer_A2C=MultimodalEncoder(self.text_model,args.layers3)
   

    self.text_linear=nn.Sequential(

        nn.Linear(args.text_size,args.text_size),

        nn.LeakyReLU()

    )

    self.image_linear=nn.Sequential(

        nn.Linear(args.image_size,args.image_size),

        nn.LeakyReLU()

    )

    self.ocr_linear=nn.Sequential(

        nn.Linear(args.text_size,args.text_size),

        nn.LeakyReLU()

    )

    self.fusion1_linear=nn.Sequential(

        nn.Linear(args.text_size,args.text_size),

        nn.LeakyReLU()

    )

    self.fusion2_linear=nn.Sequential(

        nn.Linear(args.text_size,args.text_size),

        nn.LeakyReLU()

    )

    self.fusion3_linear=nn.Sequential(

        nn.Linear(args.text_size,args.text_size),

        nn.LeakyReLU()

    )

    self.text_projection=FeatureProjection(args.text_size)

    self.visual_projection=FeatureProjection(args.image_size)

    self.ocr_projection=FeatureProjection(args.text_size)



    self.classifier_text=nn.Sequential(

        nn.Linear(args.text_size,args.num_classes,bias=True),

        nn.Tanh()

    )

    self.classifier_fusion_1=nn.Sequential(

        nn.Linear(args.text_size,args.text_size,bias=True),

        nn.LeakyReLU(),

        nn.Dropout(args.dropout_rate),

        nn.Linear(args.text_size,args.num_classes,bias=True),

        nn.Tanh()

    )

    self.classifier_fusion_2=nn.Sequential(

        nn.Linear(args.text_size,args.text_size,bias=True),

        nn.LeakyReLU(),

        nn.Dropout(args.dropout_rate),

        nn.Linear(args.text_size,args.num_classes,bias=True),

        nn.Tanh()

    )

    self.classifier_fusion_3=nn.Sequential(

        nn.Linear(args.text_size,args.text_size,bias=True),

        nn.LeakyReLU(),

        nn.Dropout(args.dropout_rate),

        nn.Linear(args.text_size,args.num_classes,bias=True),

        nn.Tanh()

    )



    self.att=nn.Linear(args.text_size,1,bias=False)

    self.loss=FocalLoss(gamma=2,alpha=[0.3, 0.3,0.2,0.2])
    # self.loss = nn.CrossEntropyLoss()

  def forward(self,inputs: dict[str,torch.tensor] ,labels=None):

    text_outputs=self.text_model(**inputs['annotation'])

    ocr_outputs=self.text_model(**inputs['ocr'])

    image_outputs=self.image_model(**inputs['image'])



    text_features=text_outputs.last_hidden_state

    text_feature=text_outputs.pooler_output



    ocr_features=ocr_outputs.last_hidden_state

    ocr_feature=ocr_outputs.pooler_output



    image_features=image_outputs.last_hidden_state

    image_feature=image_outputs.pooler_output



    text_feature=self.text_linear(text_feature)

    ocr_feature=self.ocr_linear(ocr_feature)

    image_feature=self.image_linear(image_feature)



    text_embedds=self.text_projection(text_features)

    ocr_embedds=self.ocr_projection(ocr_features)

    image_embedds=self.visual_projection(image_features)



    input_embedds_fuse1=torch.cat([image_embedds,ocr_embedds],dim=1)

    input_embedds_fuse2=torch.cat([image_embedds,text_embedds],dim=1)

    input_embedds_fuse3=torch.cat([text_embedds,ocr_embedds],dim=1)



    attention_mask_1=torch.cat((torch.ones(ocr_features.shape[0],257).to(text_features.device),inputs['ocr']['attention_mask']),dim=-1)

    attention_mask_2=torch.cat((torch.ones(text_features.shape[0],257).to(text_features.device),inputs['annotation']['attention_mask']),dim=-1)

    attention_mask_3=torch.cat((inputs['annotation']['attention_mask'],inputs['ocr']['attention_mask']),dim=-1)

    extended_attention_mask_1 = attention_mask_1.unsqueeze(1).unsqueeze(2)

    extended_attention_mask_1 = extended_attention_mask_1.to(dtype=next(self.parameters()).dtype)

    extended_attention_mask_1 = (1.0 - extended_attention_mask_1) * -10000.0

    extended_attention_mask_2 = attention_mask_2.unsqueeze(1).unsqueeze(2)

    extended_attention_mask_2 = extended_attention_mask_2.to(dtype=next(self.parameters()).dtype)

    extended_attention_mask_2 = (1.0 - extended_attention_mask_2) * -10000.0

    extended_attention_mask_3 = attention_mask_3.unsqueeze(1).unsqueeze(2)

    extended_attention_mask_3 = extended_attention_mask_3.to(dtype=next(self.parameters()).dtype)

    extended_attention_mask_3 = (1.0 - extended_attention_mask_3) * -10000.0



    fuse_hiddens_1,all_attetions_1=self.transformer_I2C(input_embedds_fuse1,extended_attention_mask_1,output_all_encoded_layers=False)

    fuse_hiddens_2,all_attetions_2=self.transformer_I2A(input_embedds_fuse2,extended_attention_mask_2,output_all_encoded_layers=False)

    fuse_hiddens_3,all_attetions_3=self.transformer_A2C(input_embedds_fuse3,extended_attention_mask_3,output_all_encoded_layers=False)

    fuse_hiddens_1=fuse_hiddens_1[-1]

    fuse_hiddens_2=fuse_hiddens_2[-1]

    fuse_hiddens_3=fuse_hiddens_3[-1]



    new_text_features_1=fuse_hiddens_2[:,257:,:]

    new_text_features_2=fuse_hiddens_3[:,:text_features.shape[1],:]

      

    new_ocr_features_1=fuse_hiddens_1[:,257:,:]

    new_ocr_features_2=fuse_hiddens_3[:,text_features.shape[1]:,:]

    

    new_image_feature_1=fuse_hiddens_1[:,0,:].squeeze(1)

    new_image_feature_2=fuse_hiddens_2[:,0,:].squeeze(1)



    new_text_feature_1 = new_text_features_1[

            torch.arange(new_text_features_1.shape[0], device=inputs['annotation']['input_ids'].device), inputs['annotation']['input_ids'].to(torch.int).argmax(dim=-1)

        ]

    new_text_feature_2 = new_text_features_2[

            torch.arange(new_text_features_2.shape[0], device=inputs['annotation']['input_ids'].device), inputs['annotation']['input_ids'].to(torch.int).argmax(dim=-1)

        ]

    new_ocr_feature_1=new_ocr_features_1[

            torch.arange(new_ocr_features_1.shape[0], device=inputs['ocr']['input_ids'].device), inputs['ocr']['input_ids'].to(torch.int).argmax(dim=-1)

        ]

    new_ocr_feature_2=new_ocr_features_2[

            torch.arange(new_ocr_features_2.shape[0], device=inputs['ocr']['input_ids'].device), inputs['ocr']['input_ids'].to(torch.int).argmax(dim=-1)

        ]



    text_weight_1=self.att(new_text_feature_1)

    text_weight_2=self.att(new_text_feature_2)

      

    ocr_weight_1=self.att(new_ocr_feature_1)

    ocr_weight_2=self.att(new_ocr_feature_2)

      

    image_weight_1=self.att(new_image_feature_1)

    image_weight_2=self.att(new_image_feature_2)



    att1=nn.functional.softmax(torch.stack((ocr_weight_1, image_weight_1), dim=-1),dim=-1)

    att2=nn.functional.softmax(torch.stack((image_weight_2, text_weight_1), dim=-1),dim=-1)

    att3=nn.functional.softmax(torch.stack((ocr_weight_2, text_weight_2), dim=-1),dim=-1)



    ow1, iw1 = att1.split([1,1], dim=-1)

    iw2,tw1= att2.split([1,1], dim=-1)

    ow2,tw2=att3.split([1,1],dim=-1)



    fuse_feature_1 = ow1.squeeze(1) * new_ocr_feature_1 + iw1.squeeze(1) * new_image_feature_1

    fuse_feature_2 = iw2.squeeze(1) * new_image_feature_2  + tw1.squeeze(1) * new_text_feature_1

    fuse_feature_3=ow2.squeeze(1)* new_ocr_feature_2 + tw2.squeeze(1)* new_text_feature_2

    

    fuse_feature_1=self.fusion1_linear(fuse_feature_1)

    fuse_feature_2=self.fusion2_linear(fuse_feature_2)

    fuse_feature_3=self.fusion2_linear(fuse_feature_3)


    fuse_concat_1 = fuse_feature_1 + image_feature + ocr_feature
    fuse_concat_2 = fuse_feature_2 + image_feature + text_feature
    fuse_concat_3 = fuse_feature_3 + text_feature + ocr_feature

    

    text_logits=self.classifier_text(text_feature)

    fusion_logits_1=self.classifier_fusion_1(fuse_concat_1)

    fusion_logits_2=self.classifier_fusion_2(fuse_concat_2)

    fusion_logits_3=self.classifier_fusion_3(fuse_concat_3)



    text_score=nn.functional.softmax(text_logits,dim=-1)

    fusion_score_1=nn.functional.softmax(fusion_logits_1,dim=-1)

    fusion_score_2=nn.functional.softmax(fusion_logits_2,dim=-1)

    fusion_score_3=nn.functional.softmax(fusion_logits_3,dim=-1)

    score=text_score+fusion_score_1+fusion_score_2+fusion_score_3

    output=score

    if labels is not None:

        loss_text=self.loss(text_logits,labels)

        loss_fusion_1=self.loss(fusion_logits_1,labels)

        loss_fusion_2=self.loss(fusion_logits_2,labels)

        loss_fusion_3=self.loss(fusion_logits_3,labels)

        loss=loss_text+loss_fusion_1+loss_fusion_2+loss_fusion_3

        output=(loss,score)

    return output