import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchinfo import summary
from pytorch_wavelets import DWTForward


class DepthSCNN_Block(nn.Module):
    def __init__(self, in_chs, emb_chs, out_chs, bn = False):
        super().__init__()
        self.block=nn.Sequential()
        self.block.add_module('embedding',nn.Conv2d(in_chs,emb_chs,kernel_size=1,padding=0,stride=1))
        in_chs = emb_chs
        if bn: self.block.add_module('bn0',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu0',nn.LeakyReLU())
        self.block.add_module('depth_conv1',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn1',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu1',nn.LeakyReLU())
        self.block.add_module('depth_conv2',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn2',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu2',nn.LeakyReLU())
        self.block.add_module('depth_conv3',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn3',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu3',nn.LeakyReLU())
        self.block.add_module('point_conv1',nn.Conv2d(in_chs,out_channels=128, kernel_size=1, padding=0, stride=1, groups=1))
        if bn: self.block.add_module('bn4',nn.BatchNorm2d(128))
        self.block.add_module('relu4',nn.LeakyReLU())
        self.block.add_module('point_conv2',nn.Conv2d(128, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1))
        if bn: self.block.add_module('bn5',nn.BatchNorm2d(64))
        self.block.add_module('relu5',nn.LeakyReLU())
        self.block.add_module('point_conv3',nn.Conv2d(64, out_channels=out_chs, kernel_size=1, padding=0, stride=1, groups=1))
        
    def forward(self, x):
        x = self.block(x)
        return x

class DPFBlock(nn.Module):
    ##outchannel = growth_rate
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
        return layer_output

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
            self.bn1 = nn.BatchNorm2d(self.in_chs)
            self.relu1 = nn.LeakyReLU()
            self.conv3x3 = nn.Conv2d(self.in_chs, self.growth_rate, kernel_size=3, padding=1, stride=1)
        else: 
            self.bottle_size = self.bottle_neck*self.growth_rate
            self.bn1 = nn.BatchNorm2d(self.in_chs)
            self.relu1 = nn.LeakyReLU()
            self.conv1x1 = nn.Conv2d(self.in_chs, self.bottle_size, kernel_size=1, padding=0, stride=1)
            self.bn2 = nn.BatchNorm2d(self.bottle_size)
            self.relu2 = nn.LeakyReLU()
            self.conv3x3 = nn.Conv2d(self.bottle_size, self.growth_rate, kernel_size=3, padding=1, stride=1)
            

    def forward(self, x):
        # x = torch.cat(feature, dim=1)
        if self.bottle_neck == 0:
            x = self.conv3x3(self.relu1(self.bn1(x)))
        else:
            x = self.conv1x1(self.relu1(self.bn1(x)))
            x = self.conv3x3(self.relu2(self.bn2(x)))
        return x
        
class classifier(nn.Module):
    def __init__(self, in_chs, num_classes, bn = False):
        super().__init__()
        self.in_chs = in_chs
        self.num_classes = num_classes
        self.bn = bn
        self.conv3x3_1 = nn.Conv2d(self.in_chs, 64, 3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv1x1_1 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(32, num_classes, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.sigm3 = nn.Sigmoid()
        self.GlobalAvgPooling= nn.AdaptiveAvgPool2d(1)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(32)

            
    def forward(self, x):
        if self.bn:
            x = self.relu1(self.bn1(self.conv3x3_1(x)))
            x = self.relu2(self.bn2(self.conv3x3_2(x)))
            x = self.sigm3(self.bn3(self.conv1x1_1(x)))
        else:
            x = self.relu1(self.conv3x3_1(x))
            x = self.relu2(self.conv3x3_2(x))
            x = self.sigm3(self.conv1x1_1(x))
        x = self.conv1x1_2(x)
        x = self.GlobalAvgPooling(x)
        return x.squeeze(-1).squeeze(-1)

class MFFWCNN(nn.Module):
 ##DepthSCNN_Block + DPFBlock + classifier
    def __init__(self, in_chs, emb_chs, SSFout_chs, num_class, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4],block_num = 1):
        super(MFFWCNN, self).__init__()
        self.wavelet = wavelet
        self.bn=bn
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck
        self.num_class = num_class
        self.SSFout_chs = SSFout_chs
        # self.embedding = nn.Conv2d(in_chs, emb_chs,kernel_size=1,padding=0,stride=1)
        # if self.bn:
        #     self.BN1 = nn.BatchNorm2d(emb_chs)
        # self.relu1 = nn.ReLU()
        self.DWT=DWTForward(J=1, wave=self.wavelet, mode=WTpadding_mode)
        self.SSFBlock_LL = DepthSCNN_Block(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_HL = DepthSCNN_Block(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_LH = DepthSCNN_Block(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_HH = DepthSCNN_Block(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.DPFBlock = DPFBlock(SSFout_chs, self.growth_rate, self.bottle_neck)
        self.classifier = classifier(self.growth_rate, self.num_class,bn=self.bn)
        self.apply(self.weight_init)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def forward(self, x):
        x_LL,x_H = self.DWT(x) ##(Batch x Channels x Width x Height)
        x_dwt = [] ##[LL, LH, HL,HH]
        x_dwt.append(self.SSFBlock_LL(x_LL))
        x_dwt.append(self.SSFBlock_LH(x_H[0][:,:,0,...]))
        x_dwt.append(self.SSFBlock_HL(x_H[0][:,:,1,...]))
        x_dwt.append(self.SSFBlock_HH(x_H[0][:,:,2,...]))
        x = self.DPFBlock(*x_dwt)
        x = self.classifier(x)
        return x


class DepthSCNN_Blockv2(nn.Module):
    ##pppddd
    def __init__(self, in_chs, emb_chs, out_chs, bn = False):
        super().__init__()
        self.block=nn.Sequential()
        self.block.add_module('transitionlayer',nn.Conv2d(in_chs,emb_chs,kernel_size=1,padding=0,stride=1))
        in_chs = emb_chs
        
        if bn: self.block.add_module('bn3',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu3',nn.LeakyReLU())
        self.block.add_module('point_conv1',nn.Conv2d(in_chs,out_channels=128, kernel_size=1, padding=0, stride=1, groups=1))
        if bn: self.block.add_module('bn4',nn.BatchNorm2d(128))
        self.block.add_module('relu4',nn.LeakyReLU())
        self.block.add_module('point_conv2',nn.Conv2d(128, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1))
        if bn: self.block.add_module('bn5',nn.BatchNorm2d(64))
        self.block.add_module('relu5',nn.LeakyReLU())
        self.block.add_module('point_conv3',nn.Conv2d(64, out_channels=out_chs, kernel_size=1, padding=0, stride=1, groups=1))
        
        in_chs = out_chs
        if bn: self.block.add_module('bn0',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu0',nn.LeakyReLU())
        self.block.add_module('depth_conv1',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn1',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu1',nn.LeakyReLU())
        self.block.add_module('depth_conv2',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn2',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu2',nn.LeakyReLU())
        self.block.add_module('depth_conv3',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        
    def forward(self, x):
        x = self.block(x)
        return x


class DepthSCNN_Blockv3(nn.Module):
    ##pdpdpd
    
    def __init__(self, in_chs, emb_chs, out_chs, bn = False):
        super().__init__()
        self.block=nn.Sequential()
        self.block.add_module('embedding',nn.Conv2d(in_chs,emb_chs,kernel_size=1,padding=0,stride=1))
        in_chs = emb_chs
        
        if bn: self.block.add_module('bn3',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu3',nn.LeakyReLU())
        self.block.add_module('point_conv1',nn.Conv2d(in_chs,out_channels=128, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = 128
        if bn: self.block.add_module('bn0',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu0',nn.LeakyReLU())
        self.block.add_module('depth_conv1',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        
        if bn: self.block.add_module('bn4',nn.BatchNorm2d(128))
        self.block.add_module('relu4',nn.LeakyReLU())
        self.block.add_module('point_conv2',nn.Conv2d(128, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = 64
        if bn: self.block.add_module('bn1',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu1',nn.LeakyReLU())
        self.block.add_module('depth_conv2',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        
        if bn: self.block.add_module('bn5',nn.BatchNorm2d(64))
        self.block.add_module('relu5',nn.LeakyReLU())
        self.block.add_module('point_conv3',nn.Conv2d(64, out_channels=out_chs, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = out_chs
        if bn: self.block.add_module('bn2',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu2',nn.LeakyReLU())
        self.block.add_module('depth_conv3',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        
    def forward(self, x):
        x = self.block(x)
        return x


class DepthSCNN_Blockv4(nn.Module):
    ##dpdpdp
    def __init__(self, in_chs, emb_chs, out_chs, bn = False):
        super().__init__()
        self.block=nn.Sequential()
        self.block.add_module('embedding',nn.Conv2d(in_chs,emb_chs,kernel_size=1,padding=0,stride=1))
        in_chs = emb_chs
        
        if bn: self.block.add_module('bn0',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu0',nn.LeakyReLU())
        self.block.add_module('depth_conv1',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn3',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu3',nn.LeakyReLU())
        self.block.add_module('point_conv1',nn.Conv2d(in_chs,out_channels=128, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = 128
        
        
        if bn: self.block.add_module('bn1',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu1',nn.LeakyReLU())
        self.block.add_module('depth_conv2',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn4',nn.BatchNorm2d(128))
        self.block.add_module('relu4',nn.LeakyReLU())
        self.block.add_module('point_conv2',nn.Conv2d(128, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = 64
        
        if bn: self.block.add_module('bn2',nn.BatchNorm2d(in_chs))
        self.block.add_module('relu2',nn.LeakyReLU())
        self.block.add_module('depth_conv3',nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, stride=1, groups=in_chs))
        if bn: self.block.add_module('bn5',nn.BatchNorm2d(64))
        self.block.add_module('relu5',nn.LeakyReLU())
        self.block.add_module('point_conv3',nn.Conv2d(64, out_channels=out_chs, kernel_size=1, padding=0, stride=1, groups=1))
        in_chs = out_chs
        
        
    def forward(self, x):
        x = self.block(x)
        return x





## outchannel = (input + growth_rate)*4
class DPFBlockv2(nn.Module):
    
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
            # layer_output=layer(*layer_input)
            layer_output=layer(torch.cat(layer_input, dim = 1))
            if idx == self.layercount-1: break
            layer_input.append(layer_output)
            layer_input.append(x_dwt[idx+1])
        layer_input.append(layer_output)
        layer_input = torch.cat(layer_input, dim = 1)
        return layer_input


##conv1x1 + conv3x3 + conv1x1 + GAP
class classifierv2(nn.Module):
    
    def __init__(self, in_chs, num_classes, bn = False):
        super().__init__()
        self.in_chs = in_chs
        self.num_classes = num_classes
        self.bn = bn
        self.conv1x1_1 = nn.Conv2d(self.in_chs, self.in_chs//2, 1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(self.in_chs//2, self.in_chs//4, 3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(self.in_chs//4, num_classes, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU()
        self.sigm = nn.Sigmoid()
        self.GlobalAvgPooling= nn.AdaptiveAvgPool2d(1)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(self.in_chs//2)
            self.bn2 = nn.BatchNorm2d(self.in_chs//4)
             
    def forward(self, x):
        if self.bn:
            x = self.relu1(self.bn1(self.conv1x1_1(x)))
            x = self.sigm(self.bn2(self.conv3x3_1(x)))
        else:
            x = self.relu1(self.conv1x1_1(x))
            x = self.sigm(self.conv3x3_1(x))
        x = self.conv1x1_2(x)
        x = self.GlobalAvgPooling(x)
        return x.squeeze(-1).squeeze(-1)



##SSFBlockv2 + DPFblockv2
class MFFBlock(nn.Module):
    
    def __init__(self, in_chs, emb_chs, SSFout_chs, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4]):
        super(MFFBlock, self).__init__()
        self.wavelet = wavelet
        self.bn=bn
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck
        self.in_chs = in_chs
        self.SSFout_chs = SSFout_chs
        self.DWT=DWTForward(J=1, wave=self.wavelet, mode=WTpadding_mode)
        self.SSFBlock_LL = DepthSCNN_Blockv2(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_HL = DepthSCNN_Blockv2(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_LH = DepthSCNN_Blockv2(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.SSFBlock_HH = DepthSCNN_Blockv2(in_chs, emb_chs, SSFout_chs, bn=self.bn)
        self.DPFBlock = DPFBlockv2(self.SSFout_chs, self.growth_rate, self.bottle_neck)
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
        x_dwt = [] ##[LL, LH, HL,HH]
        x_dwt.append(self.SSFBlock_LL(x_LL))
        x_dwt.append(self.SSFBlock_LH(x_H[0][:,:,0,...]))
        x_dwt.append(self.SSFBlock_HL(x_H[0][:,:,1,...]))
        x_dwt.append(self.SSFBlock_HH(x_H[0][:,:,2,...]))
        x = self.DPFBlock(*x_dwt)
        return x


##MFFBlock + classifier
class MFFWCNNv2(nn.Module):

    def __init__(self, in_chs, emb_chs, SSFout_chs, num_class, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFWCNNv2, self).__init__()
        self.wavelet = wavelet
        self.bn=bn
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck
        self.num_class = num_class
        # self.SSFout_chs = SSFout_chs
        self.block_num = block_num
        self.MFFBlock_list = nn.Sequential()
        out_chs = in_chs
        for i in range(block_num):
            MFFB = MFFBlock(out_chs, emb_chs, SSFout_chs, self.wavelet, WTpadding_mode, self.bn, self.growth_rate,self.bottle_neck)
            self.MFFBlock_list.add_module('MFFBlock_{}'.format(i+1), MFFB)
            out_chs = MFFB.get_final_channels(MFFB)
        
        self.classifier = classifier(out_chs, self.num_class,bn=self.bn)
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.MFFBlock_list(x)
        x = self.classifier(x)
        return x


##MFFBlock + classifierv2
class MFFWCNNv3(nn.Module):

    def __init__(self, in_chs, emb_chs, SSFout_chs, num_class, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFWCNNv3, self).__init__()
        self.wavelet = wavelet
        self.bn=bn
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck
        self.num_class = num_class
        # self.SSFout_chs = SSFout_chs
        self.block_num = block_num
        self.MFFBlock_list = nn.Sequential()
        out_chs = in_chs
        for i in range(block_num):
            MFFB = MFFBlock(out_chs, emb_chs*(i+1), SSFout_chs*(i+1),self.wavelet,WTpadding_mode,self.bn,self.growth_rate,self.bottle_neck)
            self.MFFBlock_list.add_module('MFFBlock_{}'.format(i+1), MFFB)
            out_chs = MFFB.get_final_channels(MFFB)
        
        self.classifier = classifierv2(out_chs, self.num_class,bn=self.bn)
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.MFFBlock_list(x)
        x = self.classifier(x)
        return x



##multi-Scale  classifer for each scalelayer output, consisting of conv and GAP
class classifierv3(nn.Module):
    def __init__(self, in_chs, out_num, bn = False, block_num = 1):
        super().__init__()
        self.in_chs = in_chs
        self.bn = bn
        tmpchs = 64*(2**(block_num-1))
        self.conv1x1_1 = nn.Conv2d(self.in_chs, tmpchs, 1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(tmpchs, tmpchs, 3, stride=1, padding=1, )
        self.conv1x1_2 = nn.Conv2d(tmpchs, out_num, 3, stride=1, padding=1)
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
            x = self.relu3(self.bn3(self.conv1x1_2(x)))
        else:
            x = self.relu1(self.conv1x1_1(x))
            x = self.relu2(self.conv3x3_1(x))
            x = self.relu3(self.conv1x1_2(x))
        x = self.GlobalAvgPooling(x)
        return x.squeeze(-1).squeeze(-1)

##multi-Scale  classifer for each scalelayer output, consisting of conv and GAP
class classifierv4(nn.Module):
    def __init__(self, in_chs, out_num, bn = False, block_num = 1):
        super().__init__()
        self.in_chs = in_chs
        self.bn = bn
        self.fc = nn.Linear(in_chs, out_num)
        self.relu1 = nn.LeakyReLU()
        # self.relu2 = nn.LeakyReLU()
        # self.relu3 = nn.LeakyReLU()
        self.GlobalAvgPooling= nn.AdaptiveAvgPool2d(1)
        # if self.bn:
        #     self.bn1 = nn.BatchNorm2d(in_chs)
        #     self.bn2 = nn.BatchNorm2d(64*block_num)
        #     self.bn3 = nn.BatchNorm2d(out_num)
            
    def forward(self, x):
        x = self.GlobalAvgPooling(x)
        
        # if self.bn:
        #     x = self.relu1(x)
        # else:
        x = self.fc(self.relu1(x.squeeze(-1).squeeze(-1)))
        #     x = self.relu2(self.conv3x3_1(x))
        #     x = self.relu3(self.conv1x1_2(x))
        
        return x

##Multiscale
class MFFWCNNv4(nn.Module):

    def __init__(self, in_chs, emb_chs, SSFout_chs, GAPout_num,num_class, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFWCNNv4, self).__init__()
        self.wavelet = wavelet
        self.bn=bn
        self.growth_rate = growth_rate
        self.bottle_neck = bottle_neck
        self.num_class = num_class
        # self.SSFout_chs = SSFout_chs
        self.block_num = block_num
        self.MFFBlock_list = nn.Sequential()
        self.classifier_list = nn.Sequential()
        out_chs = in_chs
        for i in range(block_num):
            MFFB = MFFBlock(out_chs, emb_chs*(i+1), SSFout_chs*(i+1), self.wavelet, WTpadding_mode, self.bn, self.growth_rate, self.bottle_neck)
            self.MFFBlock_list.add_module('MFFBlock_{}'.format(i+1), MFFB)
            out_chs = MFFB.get_final_channels(MFFB)
            self.classifier_list.add_module('classifier_{}'.format(i+1),
                                            classifierv3(out_chs, GAPout_num, bn = self.bn))
            
        # print('outchannel:',out_chs)
        self.fc1 = nn.Linear(block_num*GAPout_num, GAPout_num)
        self.fc2 = nn.Linear(GAPout_num, num_class)
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        x_scale_out = []
        for idx, (iMFFBlock, iclassifier)in enumerate(zip(self.MFFBlock_list, self.classifier_list)):
            x = iMFFBlock(x)
            y = iclassifier(x)
            x_scale_out.append(y)   
        x = torch.cat(x_scale_out, dim=1)
        x = self.fc2(torch.sigmoid(self.fc1(F.relu(x))))
        return x



class DepthSCNN_Blockv5(nn.Module):
    def __init__(self, in_chs, cmp_rate, layer_num = 1, bn = False, block_num = 1):
        super().__init__()
        self.block=nn.Sequential()
        emb_chs = int(cmp_rate*in_chs)
        if block_num > 1:
            self.block.add_module('transition layer',
                                  nn.Conv2d(in_chs, emb_chs, kernel_size=1, padding=0, stride=1))
            if bn: self.block.add_module('Tbn',nn.BatchNorm2d(emb_chs))
            self.block.add_module('Trelu',nn.LeakyReLU())
        else: emb_chs = in_chs
        for i in range(layer_num):
            self.block.add_module('depth_conv{}'.format(i+1),
                                  nn.Conv2d(emb_chs, emb_chs, kernel_size=3, padding=1, stride=1, groups=emb_chs))
            if bn: self.block.add_module('Dbn{}'.format(i+1),nn.BatchNorm2d(emb_chs))
            self.block.add_module('Drelu{}'.format(i+1),nn.LeakyReLU())
            self.block.add_module('point_conv{}'.format(i+1),
                                  nn.Conv2d(emb_chs,emb_chs,kernel_size=1,padding=0,stride=1))
            if bn: self.block.add_module('Pbn{}'.format(i+1),nn.BatchNorm2d(emb_chs))
            self.block.add_module('Prelu{}'.format(i+1),nn.LeakyReLU())
            in_chs = emb_chs
        
    def forward(self, x):
        x = self.block(x)
        return x


##multi-Scale
class MFFBlockv2(nn.Module):
    
    def __init__(self, in_chs, cmp_rate, DSClayer_num=1, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFBlockv2, self).__init__()
        self.in_chs = in_chs
        if block_num > 1:
            out_chs = int(in_chs*cmp_rate)
        else: out_chs = in_chs
        self.block_num = block_num
        self.DWT=DWTForward(J=1, wave=wavelet, mode=WTpadding_mode)
        self.SSFBlock_LL = DepthSCNN_Blockv5(in_chs, cmp_rate, DSClayer_num, bn=bn, block_num = block_num)
        self.SSFBlock_HL = DepthSCNN_Blockv5(in_chs, cmp_rate, DSClayer_num, bn=bn, block_num = block_num)
        self.SSFBlock_LH = DepthSCNN_Blockv5(in_chs, cmp_rate, DSClayer_num, bn=bn, block_num = block_num)
        self.SSFBlock_HH = DepthSCNN_Blockv5(in_chs, cmp_rate, DSClayer_num, bn=bn, block_num = block_num)
        self.DPFBlock = DPFBlockv2(out_chs, growth_rate, bottle_neck)
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
        x_dwt = [] ##[LL, LH, HL,HH]
        x_dwt.append(self.SSFBlock_LL(x_LL))
        x_dwt.append(self.SSFBlock_LH(x_H[0][:,:,0,...]))
        x_dwt.append(self.SSFBlock_HL(x_H[0][:,:,1,...]))
        x_dwt.append(self.SSFBlock_HH(x_H[0][:,:,2,...]))
        x = self.DPFBlock(*x_dwt)
        return x


##Multiscale
class MFFWCNNv5(nn.Module):

    def __init__(self, in_chs, GAPout_num, num_class, cmp_rate = 0.5, DSClayer_num = 1, wavelet='haar', WTpadding_mode = 'periodization',bn=False, growth_rate=64, bottle_neck=[0,0,4,4], block_num = 1):
        super(MFFWCNNv5, self).__init__()
        
        self.MFFBlock_list = nn.Sequential()
        self.classifier_list = nn.Sequential()
        out_chs = in_chs
        outsum = 0
        for i in range(block_num):
            MFFB = MFFBlockv2(out_chs, cmp_rate, DSClayer_num, wavelet, WTpadding_mode, bn, growth_rate, bottle_neck, i + 1)
            self.MFFBlock_list.add_module('MFFBlock_{}'.format(i+1), MFFB)
            out_chs = MFFB.get_final_channels(MFFB)
            self.classifier_list.add_module('classifier_{}'.format(i+1),
                                            classifierv3(out_chs, GAPout_num, bn = bn, block_num = i+1))
            outsum += out_chs
            
        # print('outchannel:',out_chs)
        self.fc1 = nn.Linear(GAPout_num*block_num, num_class)
        # self.fc1 = nn.Linear(outsum, num_class)
        # self.fc2 = nn.Linear(GAPout_num, num_class)
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)
    
    def forward(self, x):
        x_scale_out = []
        for idx, (iMFFBlock, iclassifier)in enumerate(zip(self.MFFBlock_list, self.classifier_list)):
            x = iMFFBlock(x)
            y = iclassifier(x)
            x_scale_out.append(y)   
        x = torch.cat(x_scale_out, dim=1)
        # x = self.fc2(torch.sigmoid(self.fc1(F.relu(x))))
        x = self.fc1(F.leaky_relu(x))
        return x

if __name__ == '__main__':
    
    # model = MFFWCNNv4(102,24,24,64,9,wavelet='haar',WTpadding_mode ='zero', growth_rate=12,bottle_neck=[2,2,2,2],bn=True,block_num = 3)
    model = MFFWCNNv5(15,64,9,cmp_rate=0.5,DSClayer_num=3,wavelet='haar',WTpadding_mode ='zero', growth_rate=12,bottle_neck=[2,2,2,2],bn=True,block_num = 3)
    model.train()
    x=torch.randn(64,15,24,24)
    target=torch.randint(0,9,(64,))
    y=model(x)
    criterion = nn.CrossEntropyLoss()
    out=criterion(y,target)
    summary(model, (10,15,24,24),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=15,row_settings=['var_names'],depth=3)
