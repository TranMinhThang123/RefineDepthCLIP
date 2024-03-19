# Hyperparameter Control:
depth_templates = ['This {} is {}'] 
obj_classes=['object']
depth_classes =['giant', 'extremely close', 'close','not in distance','a little remote', 'far','unseen'] 
bin_list=[1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature=0.1
clip_vis = 'RN50'


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    #print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from torch.jit import script
import geffnet
import clip


def zeroshot_classifier(depth_classes,obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj,depth) for template in templates]  # format with class
                texts = clip.tokenize(texts) # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


class AdapterLayer(nn.Module):
    def __init__(self, c_in, reduction=4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in,int(c_in//reduction)),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = x.permute(1,0)
        x = self.fc(x)
        x = x.permute(1,0)

        return x


# CLIP for Monocular Depth Estimation
class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.ResNet = self.clip.visual
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip) # init text feature

        self.text_f.requires_grad = False
        for param in self.ResNet.parameters():
            param.require_grad = False

        self.adapter1 = AdapterLayer(c_in=1024).to(self.clip.dtype)
        self.adapter2 = AdapterLayer(c_in=1024,reduction=2)
        self.adapter3 = AdapterLayer(c_in=1024,reduction=1)
        self.adapter4 = AdapterLayer(c_in=1024,reduction=1/2)

        self.bin_list1 = nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1)
        self.bin_list2 = nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1)
        self.bin_list3 = nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1)
        self.bin_list4 = nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1)



    def forward(self, x):
        print("*"*150)
        # img_f = self.clip.encode_image(x).reshape(1,2048,300).permute(0,2,1) # B, HW, C
        # img_f = img_f / img_f.norm(dim=-1, keepdim=True) # normalize img_f
        print(self.text_f.shape)
        def stem(x):
            for conv, bn in [(self.ResNet.conv1, self.ResNet.bn1), (self.ResNet.conv2, self.ResNet.bn2), (self.ResNet.conv3, self.ResNet.bn3)]:
                x = self.ResNet.relu(bn(conv(x)))
            x = self.ResNet.avgpool(x)
            return x

        x = stem(x)

        feature_map1 = self.ResNet.layer1(x)
        feature_map2 = self.ResNet.layer2(feature_map1)
        feature_map3 = self.ResNet.layer3(feature_map2)
        feature_map4 = self.ResNet.layer4(feature_map3)
        print(feature_map1.shape,end="=>")
        print(feature_map2.shape,end="=>")
        print(feature_map3.shape,end="=>")
        print(feature_map4.shape)

        feature_map1 = feature_map1.reshape(1,256,120*160)
        feature_map1 = feature_map1.permute(0,2,1)
        feature_map1 = feature_map1/feature_map1.norm(dim=-1,keepdim=True)
        prompt1 = self.adapter1(self.text_f)
        print(prompt1.size())
        depth_map1 = feature_map1@prompt1
        depth_map1 = depth_map1.reshape(-1,self.bins,120,160)
        depth_map1 = F.softmax(depth_map1,dim=1)
        print("depth map 1 shape: ",depth_map1.shape)
        print("feature map 1 shape: ",feature_map1.shape)

        feature_map2 = feature_map2.reshape(1,512,60*80)
        feature_map2 = feature_map2.permute(0,2,1)
        feature_map2 = feature_map2/feature_map2.norm(dim=-1,keepdim=True)
        prompt2 = self.adapter2(self.text_f)
        print(prompt2.size())
        depth_map2 = feature_map2@prompt2
        depth_map2 = depth_map2.reshape(-1,self.bins,60,80)
        depth_map2 = F.softmax(depth_map2,dim=1)
        print("depth map 2 shape: ",depth_map2.shape)
        print("feature map 2 shape: ",feature_map2.shape)


        feature_map3 = feature_map3.reshape(1,1024,30*40)
        feature_map3 = feature_map3.permute(0,2,1)
        feature_map3 = feature_map3/feature_map3.norm(dim=-1,keepdim=True)
        prompt3 = self.adapter3(self.text_f)
        print(prompt3.size())
        depth_map3 = feature_map3@prompt3
        depth_map3 = depth_map3.reshape(-1,self.bins,30,40)
        depth_map3 = F.softmax(depth_map3,dim=1)
        print("depth map 3 shape: ",depth_map3.shape)
        print("feature map 3 shape: ",feature_map3.shape)

        feature_map4 = feature_map4.reshape(1,2048,15*20)
        feature_map4 = feature_map4.permute(0,2,1)
        feature_map4 = feature_map4/feature_map4.norm(dim=-1,keepdim=True)
        prompt4 = self.adapter4(self.text_f)
        print(prompt4.size())
        depth_map4 = feature_map4@prompt4
        depth_map4 = depth_map4.reshape(-1,self.bins,15,20)
        depth_map4 = F.softmax(depth_map4,dim=1)
        depth_map4 = depth_map4*self.bin_list4
        print("depth map 4 shape: ",depth_map4.shape)
        print("feature map 4 shape: ",feature_map4.shape)

        return img_f


