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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zeroshot_classifier(depth_classes,obj_classes, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for depth in depth_classes:
            for obj in obj_classes:
                texts = [template.format(obj,depth) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(device) # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


class AdapterLayer(nn.Module):
    def __init__(self, c_in, reduction=4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in,int(c_in//reduction)).to(device).to(torch.float32),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = x.permute(1,0)
        x = self.fc(x)
        x = x.permute(1,0)

        return x
    


class Conv2DLayerBlock(nn.Module):
    def __init__(self,in_channel=14,out_channel=7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1).to(device),
            nn.BatchNorm2d(7).to(device=device),
            nn.ReLU(inplace=True).to(device=device)
        )

    def forward(self,x):
        x = self.conv(x)
        return x



# CLIP for Monocular Depth Estimation
class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip).to(torch.float32) # init text feature
        
        
        self.upsample_layer = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        self.conv_block = [Conv2DLayerBlock() for _ in range(3)]
        # self.last_conv_layer = nn.Sequential(
        #     nn.Conv2d(7,1,kernel_size=1,stride=1,padding=0).to(device),
        #     nn.ReLU(inplace=True).to(device)
        # )

        self.text_f.requires_grad = False
        for param in self.clip.visual.parameters():
            param.requires_grad = False

        self.adapter_list = [AdapterLayer(c_in=1024, reduction=4/(2**i)) for i in range(4)]
        # self.bin_list = [nn.Parameter(torch.rand(1,7),requires_grad=True).unsqueeze(-1).unsqueeze(-1).to(device) for _ in range(4)]
        self.bin_depth = torch.tensor(bin_list).unsqueeze(-1).unsqueeze(-1).to(device)
        self.size_list = [(120,160),(60,80),(30,40),(15,20)]
        self.channel_list = [256,512,1024,2048]


    def compute_depth_map(self,x):
        batch_size = x.shape[0]

        def stem(x):
            for conv, bn in [(self.clip.visual.conv1, self.clip.visual.bn1), (self.clip.visual.conv2, self.clip.visual.bn2), (self.clip.visual.conv3, self.clip.visual.bn3)]:
                x = self.clip.visual.relu(bn(conv(x)))
            x = self.clip.visual.avgpool(x)
            return x


        x = x.type(self.clip.visual.conv1.weight.dtype)
        x = stem(x)

        feature_map1 = self.clip.visual.layer1(x)
        feature_map2 = self.clip.visual.layer2(feature_map1)
        feature_map3 = self.clip.visual.layer3(feature_map2)
        feature_map4 = self.clip.visual.layer4(feature_map3)

        feature_map_list = [feature_map1.to(torch.float32),feature_map2.to(torch.float32),feature_map3.to(torch.float32),feature_map4.to(torch.float32)]
        feature_map_list = [feature_map_list[i].reshape(batch_size,self.channel_list[i],self.size_list[i][0]*self.size_list[i][1]).permute(0,2,1) for i in range(4)]# B,H*W,C
        feature_map_list = [fea/fea.norm(dim=-1,keepdim=True) for fea in feature_map_list]# norm 
        prompts_list = [self.adapter_list[i](self.text_f) for i in range(4)]

        depth_map_list = [100.*feature_map_list[i]@prompts_list[i]/temperature for i in range(4)]
        depth_map_list = [depth_map_list[i].permute(0,2,1).reshape(-1,self.bins,*self.size_list[i]) for i in range(4)]
        # depth_map_list = [F.softmax(depth_map_list[i],dim=1)*self.bin_list[i] for i in range(4)]
        
        return depth_map_list


    def forward(self, x):
        depth_map1,depth_map2,depth_map3,depth_map4 = self.compute_depth_map(x)
        # print("forward part")
        output = self.upsample_layer(depth_map4)
        # print("After upsample depth map 4: ",output.shape)
        output = torch.cat((output,depth_map3),dim=1)
        # print("After cat output vs depth map 3: ",output.shape)
        output = self.conv_block[0](output)
        # print("After pass through conv",output.shape)
        output = self.upsample_layer(output)
        # print("After upsample depth map 3: ",output.shape)
        output = torch.cat((output,depth_map2),dim=1)
        # print("After cat output vs depth map 2: ",output.shape)
        output = self.conv_block[1](output)
        # print("After pass through conv",output.shape)
        output = self.upsample_layer(output)
        # print("After upsample depth map 2: ",output.shape)
        output = torch.cat((output,depth_map1),dim=1)
        # print("After cat output vs depth map 1: ",output.shape)
        output = self.conv_block[2](output)
        # print("After pass through conv",output.shape)
        depth = F.softmax(output,dim=1)*self.bin_depth
        depth = depth.sum(dim=1,keepdim=True)
        # depth = self.last_conv_layer(output)
        # print("depth shape: ",depth.shape)
        depth = nn.functional.interpolate(depth,size=[480,640],mode="bilinear",align_corners=True)

        # print("depth output shape: ",depth.shape)

        return depth


