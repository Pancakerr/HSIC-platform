import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.nn.init as init


class SPEFCN(nn.Module):
    def __init__(self, in_ch):
        super(SPEFCN, self).__init__()
        
        self.spe_conv1 = nn.Sequential(
            nn.Conv2d(in_ch,64,1,bias=True),
            nn.ReLU()
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv2d(64,64,1,bias=True),
            nn.ReLU()
        )
        self.spe_conv3 = nn.Sequential(
            nn.Conv2d(64,64,1,bias=True),
            nn.ReLU()
        )
    def forward(self,x):
        x1 = self.spe_conv1(x)
        x2 = self.spe_conv2(x1)
        x3 = self.spe_conv3(x2)
        return x1+x2+x3
            
class SPAFCN(nn.Module):
    def __init__(self,in_ch):
        super(SPAFCN, self).__init__()
        self.dr_conv = nn.Sequential(
            nn.Conv2d(in_ch,64,1,bias=True),
            nn.ReLU()
        )
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(64,64,3,dilation=2,padding=2,bias=True),
            nn.ReLU(),
            nn.AvgPool2d(3,1,1)
        )
        self.spa_conv2 = nn.Sequential(
            nn.Conv2d(64,64,3,dilation=2,padding=2,bias=True),
            nn.ReLU(),
            nn.AvgPool2d(3,1,1)
        )
    def forward(self,x):
        x1 = self.dr_conv(x)
        x2 = self.spa_conv1(x1)
        x3 = self.spa_conv2(x2)
        return x1+x2+x3

class SSFCN(nn.Module):
    def __init__(self,in_ch,class_num):
        super(SSFCN, self).__init__()
        self.SPE = SPEFCN(in_ch)
        self.SPA = SPAFCN(in_ch)
        self.w1 = nn.Parameter(torch.zeros(1).uniform_(1,2))
        self.w2 = nn.Parameter(torch.zeros(1).uniform_(1,2))
        self.merge = nn.Sequential(
            nn.Conv2d(64,class_num,1,bias=True),
            nn.ReLU()
        )
        self.apply(self.weight_init)
        
    def forward(self,x):
        x_spe = self.SPE(x)
        x_spa = self.SPA(x)
        x_ss = self.w1*x_spe + self.w2*x_spa
        out = self.merge(x_ss)
        return out
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.uniform_(m.weight,0.01,0.02)
            if m.bias is not None:
                init.zeros_(m.bias)
        


if __name__ == '__main__':
    ## 103 200 
    net = SSFCN(103,9)
    summary(net, input_data=torch.randn((1,103,610,340)),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=5,device='cpu')