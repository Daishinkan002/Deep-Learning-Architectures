import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop



class BiConv_Stage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=False, pad=True) -> None:
        super(BiConv_Stage, self).__init__()
        self.pad = pad
        if(upsample):
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
    
    def forward(self, X, X2=None):
        
        if(X2 is not None):
            X = self.upsample(X)
            difference_y = X2.size()[2] - X.size()[2]
            difference_x = X2.size()[3] - X.size()[3]
            if(self.pad):
                X = F.pad(X, (difference_x//2, difference_x - difference_x//2, difference_y//2, difference_y - difference_y//2))
            else:
                centercrop = CenterCrop((X.size()[2], X.size()[3]))
                X2 = centercrop(X2)
            
            X = torch.cat([X, X2], dim=1)
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.act1(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X_out = self.act2(X)
        return X_out



class Network(nn.Module):
    def __init__(self, num_channels, num_classes, padding=False):
        super(Network, self).__init__()
        self.pad = padding
        self.downsampling_stage1 = BiConv_Stage(num_channels, 64)
        self.max1 = nn.MaxPool2d(2)
        self.downsampling_stage2 = BiConv_Stage(64, 128)
        self.max2 = nn.MaxPool2d(2)
        self.downsampling_stage3 = BiConv_Stage(128, 256)
        self.max3 = nn.MaxPool2d(2)
        self.downsampling_stage4 = BiConv_Stage(256, 512)
        self.max4 = nn.MaxPool2d(2)
        self.downsampling_stage5 = BiConv_Stage(512, 1024)
        
        self.upsampling_stage1 = BiConv_Stage(1024, 512, 2, True, self.pad)
        self.upsampling_stage2 = BiConv_Stage(512, 256, 2, True, self.pad)
        self.upsampling_stage3 = BiConv_Stage(256, 128, 2, True, self.pad)
        self.upsampling_stage4 = BiConv_Stage(128, 64, 2, True, self.pad)
        
        self.last_conv = nn.Conv2d(64, num_classes, 1)


    
    def forward(self, X):
        enc1 = self.downsampling_stage1(X)
        enc2 = self.max1(enc1)
        enc2 = self.downsampling_stage2(enc2)
        enc3 = self.max2(enc2)
        enc3 = self.downsampling_stage3(enc3)
        enc4 = self.max3(enc3)
        enc4 = self.downsampling_stage4(enc4)
        enc5 = self.max4(enc4)
        enc5 = self.downsampling_stage5(enc5)        
        
        X = self.upsampling_stage1(enc5, enc4)
        X = self.upsampling_stage2(X, enc3)
        X = self.upsampling_stage3(X, enc2)
        X = self.upsampling_stage4(X, enc1)
        X = self.last_conv(X)
        return X


num_channels = 3
num_classes = 2

model = Network(num_channels, num_classes, padding=False)

print("Model - \n", model)
img = np.ones((1,3,572,572))
img = torch.Tensor(img)
print("Final Output shape : ", model(img).shape)

        

        
