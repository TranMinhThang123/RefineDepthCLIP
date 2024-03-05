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


class FCLayer(nn.Module):
    def __init__(self, c_in=1024, reduction=4):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# CLIP for Monocular Depth Estimation
class MonoCLIP(nn.Module):
    def __init__(self):
        super(MonoCLIP, self).__init__()
        self.bins = len(depth_classes)

        self.clip, _ = clip.load(clip_vis) # load pretrained clip encoder
        self.text_f = zeroshot_classifier(depth_classes, obj_classes, depth_templates, self.clip) # init text feature

        self.adapter = FCLayer(1024).to(self.clip.dtype)


    def forward(self, x):
        img_f = self.clip.encode_image(x).permute(1, 0, 2)  # B, HW, C
        img_f = img_f / img_f.norm(dim=-1, keepdim=True) # normalize img_f

        # @: dot product of two vectors
        img_f=torch.nn.functional.interpolate(img_f,scale_factor=0.5) # to match size
        print("*"*50)
        print("img_f shape: ",img_f.shape,"text_f shape: ",self.text_f.shape)
        depth_logits = 100. * img_f @ self.text_f  # B, HW, K # img_f and text_f have both been normalized, so just use a inner product
        print("depth logit shape: ",depth_logits.shape)
        depth_logits = depth_logits.permute(0, 2, 1).reshape(-1, self.bins, 15, 20)  # B, K, H, W 
        depth_logits/=temperature

        depth = F.softmax(depth_logits, dim=1)
        bin_tensor=torch.tensor(bin_list).to(depth.device)
        depth = depth * bin_tensor.reshape(1, self.bins).unsqueeze(-1).unsqueeze(-1)
        depth = depth.sum(1, keepdim=True)
        return depth   


