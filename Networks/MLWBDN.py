import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torchinfo import summary
from pytorch_wavelets import DWTForward


class Denselayer(nn.Module):
    '''Argus:
            init_channel: channels of dense block input, also equal to channels of LL/HL/LH/HH
            num_layer:    count of denselayer, begin with 1 
            growth_rate:  out_channels of 3x3conv as output channels of layer
            bottle_neck:  out_channels of bottleneck 1x1conv = bottle_neck*growth_size,'0' means remove bottleneck layer
    '''
    
    def __init__(self, in_chs, num_layer, growth_rate, bottle_neck=0):
        super().__init__()
        self.bottle_neck = bottle_neck
        self.growth_rate = growth_rate
        self.num_layer = num_layer
        self.in_chs = in_chs
        
        if self.bottle_neck == 0:
            self.conv3x3 = nn.Conv2d(self.in_chs, self.growth_rate, kernel_size=3, padding=1, stride=1)
            self.bn1 = nn.BatchNorm2d(self.growth_rate)
            self.relu1 = nn.LeakyReLU()
        else: 
            self.bottle_size = self.bottle_neck*self.growth_rate
            self.conv1x1 = nn.Conv2d(self.in_chs, self.bottle_size, kernel_size=1, padding=0, stride=1)
            self.bn1 = nn.BatchNorm2d(self.bottle_size)
            self.relu1 = nn.LeakyReLU()
            self.conv3x3 = nn.Conv2d(self.bottle_size, self.growth_rate, kernel_size=3, padding=1, stride=1)
            self.bn2 = nn.BatchNorm2d(self.growth_rate)
            self.relu2 = nn.LeakyReLU()
            
    def forward(self, x):
        # x = torch.cat(feature, dim=1)
        if self.bottle_neck == 0:
            x = self.relu1(self.bn1(self.conv3x3(x)))
        else:
            x = self.relu1(self.bn1(self.conv1x1(x)))
            x = self.relu2(self.bn2(self.conv3x3(x)))
        return x

## outchannel = (input + growth_rate)*4
class SADFBlock(nn.Module):
    
    def __init__(self, in_chs, growth_rate, bottle_neck_list=[0,0,4,4], layercount=4):
        super().__init__()
        self.in_chs = [in_chs*(i+1)+i*growth_rate for i in range(layercount)]
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck_list
        self.layercount = layercount
        self.layer=nn.Sequential()
        for i in range(self.layercount):
            self.layer.add_module('Denslayer{}'.format(i+1),Denselayer(self.in_chs[i], i+1, self.growth_rate, self.bottle_neck[i]))
        
    def forward(self, *x_dwt):
        ##x_dwt:[LL,LH,HL,HH]
        layer_input = [x_dwt[0]] ##LL
        for idx, layer in enumerate(self.layer):
            layer_output=layer(torch.cat(layer_input, dim = 1))
            if idx == self.layercount-1: break
            layer_input.append(layer_output)
            layer_input.append(x_dwt[idx+1])
        layer_input.append(layer_output)
        layer_input = torch.cat(layer_input, dim = 1)
        return layer_input

## common denseblock
class SADFBlockv2(nn.Module):
    
    def __init__(self, in_chs, growth_rate, bottle_neck_list=[4,4,4,4], layercount=4):
        super().__init__()
        self.in_chs = [in_chs*4+i*growth_rate for i in range(layercount)]
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck_list
        self.layercount = layercount
        self.layer=nn.Sequential()
        for i in range(self.layercount):
            self.layer.add_module('Denslayer{}'.format(i+1),Denselayer(self.in_chs[i], i+1, self.growth_rate, self.bottle_neck[i]))
        
    def forward(self, *x_dwt):
        ##x_dwt:[LL,LH,HL,HH]
        layer_input = [x_dwt[0],x_dwt[1],x_dwt[2],x_dwt[3]]
        for idx, layer in enumerate(self.layer):
            layer_output=layer(torch.cat(layer_input, dim = 1))
            layer_input.append(layer_output)
        layer_input = torch.cat(layer_input, dim = 1)
        return layer_input

class classifier(nn.Module):
    def __init__(self, in_chs, out_num, bn = False, block_num = 1):
        super().__init__()
        self.in_chs = in_chs
        self.bn = bn
        tmpchs = 128
        self.dropout = nn.Dropout(p = 0.3)
        self.conv1x1_1 = nn.Conv2d(self.in_chs, tmpchs, 1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(tmpchs, tmpchs, 3, stride=1, padding=1,groups=tmpchs)
        self.conv3x3_2 = nn.Conv2d(tmpchs, out_num, 3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.GlobalAvgPooling= nn.AdaptiveAvgPool2d(1)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(tmpchs)
            self.bn2 = nn.BatchNorm2d(tmpchs)
            self.bn3 = nn.BatchNorm2d(out_num)
            
    def forward(self, x):
        if self.bn:
            x = self.relu1(self.bn1(self.conv1x1_1(x)))
            x = self.relu2(self.bn2(self.conv3x3_1(x)))
            x = self.relu3(self.bn3(self.conv3x3_2(x)))
        else:
            x = self.relu1(self.conv1x1_1(x))
            x = self.relu2(self.conv3x3_1(x))
            x = self.relu3(self.conv3x3_2(x))
        x = self.dropout(self.GlobalAvgPooling(x))
        return x.squeeze(-1).squeeze(-1)

##multi-Scale
class MFFBlock(nn.Module):
    def __init__(self, in_chs, cmp_rate, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFBlock, self).__init__()
        self.in_chs = in_chs
        if block_num == 1:
            out_chs = in_chs
        else:
            out_chs = int(in_chs*cmp_rate)
        self.Transitionlayer_LL = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU())
        self.Transitionlayer_HL = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU())
        self.Transitionlayer_LH = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU())
        self.Transitionlayer_HH = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_chs),
            nn.LeakyReLU())
        self.block_num = block_num
        self.DWT=DWTForward(J=1, wave=wavelet, mode=WTpadding_mode)
        # self.SADFBlock = SADFBlock(out_chs, growth_rate, bottle_neck)
        self.SADFBlock = SADFBlock(out_chs, growth_rate, bottle_neck)
        self.apply(self.weight_init)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    @staticmethod
    def get_final_channels(self):
        x = torch.zeros(1,self.in_chs,5,5)
        with torch.no_grad():
            x =self.forward(x)
        return x.shape[1]
    
    def forward(self, x):
        x_LL,x_H = self.DWT(x) ##(Batch x Channels x Width x Height)
        if self.block_num > 0:    
            x_dwt = [] ##[LL, LH, HL,HH]
            x_dwt.append(self.Transitionlayer_LL(x_LL))
            x_dwt.append(self.Transitionlayer_LH(x_H[0][:,:,0,...]))
            x_dwt.append(self.Transitionlayer_HL(x_H[0][:,:,1,...]))
            x_dwt.append(self.Transitionlayer_HH(x_H[0][:,:,2,...]))
        else: 
            x_dwt = [x_LL,x_H[0][:,:,0,...],x_H[0][:,:,1,...],x_H[0][:,:,2,...]]
        x = self.SADFBlock(*x_dwt)
        return x


##Multiscale
class MLWBDN(nn.Module):
    def __init__(self, in_chs, GAPout_num, num_class, cmp_rate = 0.1, wavelet='haar', WTpadding_mode = 'periodization',bn=True, growth_rate=48, bottle_neck=[4,4,4,4], block_num = 1):
        super(MLWBDN, self).__init__()
        self.MFFBlock_list = nn.Sequential()
        self.classifier_list = nn.Sequential()
        self.trans_layer_list = nn.Sequential()
        out_chs = in_chs
        trans_chs = 0
        for i in range(block_num):
            MFFB = MFFBlock(out_chs, cmp_rate, wavelet, WTpadding_mode, bn, growth_rate, bottle_neck, i + 1)
            self.MFFBlock_list.add_module('MFFBlock_{}'.format(i+1), MFFB)
            out_chs = MFFB.get_final_channels(MFFB)
            self.classifier_list.add_module('classifier_{}'.format(i+1),
                                            classifier(out_chs+trans_chs, GAPout_num, bn = bn, block_num = i+1))
            if (i+1) < block_num:
                trans_in_chs =  out_chs+trans_chs
                trans_out_chs = int((out_chs+trans_chs)*cmp_rate)
                self.trans_layer_list.add_module('translayer{}'.format(i),nn.Sequential(
                    nn.Conv2d(trans_in_chs,trans_out_chs, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(trans_out_chs),nn.LeakyReLU()))
                trans_chs = trans_out_chs
                
                
            # self.classifier_list.add_module('classifier_{}'.format(i+1),
            #                                 classifierv2(out_chs, num_class, bn = bn, block_num = i+1))
            
        self.fc1 = nn.Linear(GAPout_num*block_num, num_class)
       
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x1):
        x_scale_out = []
        
        for idx, (iMFFBlock, iclassifier)in enumerate(zip(self.MFFBlock_list, self.classifier_list)):
            x1 = iMFFBlock(x1)
            if idx == 0:
                x2 = torch.clone(x1)
            else: 
                x2 = self.trans_layer_list[idx-1](x2)
                x2 = torch.cat([x1,x2],dim=1)
            
            y = iclassifier(x2)
            x_scale_out.append(y)   
        x = torch.cat(x_scale_out, dim=1)
        x = self.fc1(F.leaky_relu(x))
        return x

if __name__ == '__main__':
    
    model = MLWBDN(15,36,9,cmp_rate=0.1,wavelet='haar', growth_rate=36,bottle_neck=[4,4,4,4],bn=True,block_num = 3)
    model.train()
    x=torch.randn(10,15,24,24)
    target=torch.randint(0,9,(10,))
    y=model(x)
    criterion = nn.CrossEntropyLoss()
    out=criterion(y,target)
    summary(model, input_data=x,col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=3)
