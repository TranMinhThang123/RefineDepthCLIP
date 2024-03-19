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
    def __init__(self, decode_type:str):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        assert decode_type in ["upsample","deconv"]," Only supper decode type as upsample or deconv "
        self.decode_type = decode_type
        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.ResNet = self.clip.visual
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip) # init text feature
        self.upsample_layer = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)

        self.text_f.requires_grad = False
        for param in self.ResNet.parameters():
            param.require_grad = False

        self.adapter_list = [AdapterLayer(c_in=1024, reduction=4/(2**i)) for i in range(4)]
        self.bin_list = [nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1) for i in range(4)]
        self.size_list = [(120,160),(60,80),(30,40),(15,20)]
        self.channel_list = [256,512,1024,2048]


    def compute_depth_map(self,x):
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


        feature_map_list = [feature_map1,feature_map2,feature_map3,feature_map4]
        feature_map_list = [feature_map_list[i].reshape(1,self.channel_list[i],self.size_list[i][0]*self.size_list[i][1]).permute(0,2,1) for i in range(4)]
        feature_map_list = [fea/fea.norm(dim=-1,keepdim=True) for fea in feature_map_list]
        prompts_list = [self.adapter_list[i](self.text_f) for i in range(4)]


        depth_map_list = [feature_map_list[i]@prompts_list[i] for i in range(4)]
        depth_map_list = [depth_map_list[i].reshape(-1,self.bins,*self.size_list[i]) for i in range(4)]
        depth_map_list = [F.softmax(depth_map_list[i],dim=1)*self.bin_list[i] for i in range(4)]
        for i in depth_map_list:
            print(i.shape)

        return depth_map_list


    def forward(self, x):
        
        depth_map1,depth_map2,depth_map3,depth_map4 = self.compute_depth_map(x)
        if self.decode_type == "upsample":
            output = self.upsample_layer(depth_map4)
            output = self.upsample_layer(torch.cat)
        else:
            pass



    

        return img_f


