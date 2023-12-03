import torch
from torch import nn



class SubtractionLayer(nn.Module):
    def __init__(self):
        super(SubtractionLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 - x2




class MultiplicationLayer(nn.Module):
    def __init__(self):
        super(MultiplicationLayer, self).__init__()
    def forward(self, x1, x2):
        return x1*x2




class ResidualBlock(nn.Module):
    def __init__(self, bn_filters, num_filter, stride, stage_name, dim_match=False):
        super(ResidualBlock, self).__init__()
        self.dim_match = dim_match
        self.stage_name = stage_name
        self.model = torch.nn.ModuleDict({
            stage_name + "_bn1" : nn.BatchNorm2d(num_filter, eps=2e-5, momentum=0.9),
            stage_name + "_conv1" : nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=False),
            stage_name + "_bn2" : nn.BatchNorm2d(num_filter, eps=2e-5, momentum=0.9),
            stage_name + "_relu1" : nn.PReLU(),
            stage_name + "_conv2" : nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=stride, padding=1, bias=False),
            stage_name + "_bn3" : nn.BatchNorm2d(num_filter, eps=2e-5, momentum=0.9)})

        # })

        if(not dim_match):
            self.model[f"{stage_name}_bn1"] = nn.BatchNorm2d(bn_filters, eps=2e-5, momentum=0.9)
            self.model[f"{stage_name}_conv1"] = nn.Conv2d(bn_filters, num_filter, kernel_size=3, stride=1, padding=1, bias=False)
        
        if not self.dim_match:
            self.model[f"{stage_name}_conv1sc"] = nn.Conv2d(bn_filters, num_filter, kernel_size=1, stride=stride, bias=False)
            self.model[f"{stage_name}_sc"] = nn.BatchNorm2d(num_filter, momentum=0.9, eps=2e-5)
        

    def forward(self, x):
        out = self.model[f"{self.stage_name}_bn1"](x)
        out = self.model[f"{self.stage_name}_conv1"](out)
        out = self.model[f"{self.stage_name}_bn2"](out)
        out = self.model[f"{self.stage_name}_relu1"](out)
        out = self.model[f"{self.stage_name}_conv2"](out)
        out = self.model[f"{self.stage_name}_bn3"](out)
        if(self.dim_match):
            shortcut = x
        else:
            shortcut = self.model[f"{self.stage_name}_conv1sc"](x)
            shortcut = self.model[f"{self.stage_name}_sc"](shortcut)

        return out + shortcut





class ResNet101(nn.Module):
    def __init__(self, bn_mom=0.9, act_type='prelu'):
        super(ResNet101, self).__init__()
        self.filter_list=[64, 64, 128, 256, 512],
        self.units = [3, 13, 30, 3]
        self.filter_list = [64, 64, 128, 256, 512]

        self.layers = nn.ModuleDict()
        self.layers['identity'] =  nn.Identity()
        self.layers['sub'] =  SubtractionLayer()
        self.layers['mul'] =  MultiplicationLayer()
        self.layers['conv0'] =  nn.Conv2d(3, self.filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layers['bn0'] =  nn.BatchNorm2d(self.filter_list[0], eps=2e-5, momentum=bn_mom)
        self.layers['act0'] =  nn.PReLU()
        for i in range(4):
            stage_name = f"stage{i+1}_unit1"
            self.layers[stage_name] = ResidualBlock(self.filter_list[i], self.filter_list[i+1], 2, stage_name, False)
            for j in range(self.units[i] -1):
                sec_stage_unit_name = f'stage{i+1}_unit{j+2}'
                self.layers[sec_stage_unit_name] = ResidualBlock(self.filter_list[i], self.filter_list[i+1], 1, sec_stage_unit_name, True)

        self.layers['bn1'] =  nn.BatchNorm2d(self.filter_list[-1], eps=2e-5, momentum=bn_mom)
        self.layers['dropout'] =  nn.Dropout(0.4)
        self.layers['flatten'] =  nn.Flatten()
        self.layers['pre_fc1'] =  nn.Linear(25088, 512)
        self.layers['fc1'] = nn.BatchNorm1d(512, eps=2e-5, momentum=bn_mom)

    def forward(self, input):
        X = self.layers['identity'](input)
        X = self.layers['sub'](X, 127.5)
        X = self.layers['mul'](X, 0.0078125)
        X = self.layers['conv0'](X)
        X = self.layers['bn0'](X)
        X = self.layers['act0'](X)


        for i in range(4):
            X = self.layers[f'stage{i+1}_unit1'](X)
            for j in range(self.units[i] - 1):
                X = self.layers[f'stage{i+1}_unit{j+2}'](X)
        
        X = self.layers['bn1'](X)
        X = self.layers['dropout'](X)
        X = self.layers['flatten'](X)
        X = self.layers['pre_fc1'](X)
        X = self.layers['fc1'](X)

        return X

