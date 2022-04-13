import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

'''
    Reference:
        Paper links: [IEEE T-GARS paper](http://ieeexplore.ieee.org/document/8061020/) and [IGARSS2017 paper](https://www.researchgate.net/publication/320145356_Deep_Residual_Networks_for_Hyperspectral_Image_Classification).
        Keras implementation code download link: [SSRN code](https://github.com/zilongzhong/SSRN/archive/master.zip).
        Pytorch implementation code: https://github.com/zilongzhong/SSTN
'''

class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()
                
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)
        #self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        input = input.unsqueeze(1)
        out = self.s1(input)
        return out.squeeze(1) 


class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)
    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(2)
        return out

class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()
                
        self.spc1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm3d(in_channels),)
        
        self.spc2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),)
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
                
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        
        return F.leaky_relu(out + input)


class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.leaky_relu(out + input)


class SSRN(nn.Module):
    def __init__(self, num_classes=9, k=49):
        super(SSRN, self).__init__()
        self.layer1 = SPCModuleIN(1, 28)
        self.layer2 = ResSPC(28,28)
        self.layer3 = ResSPC(28,28)  
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)
        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):

        x = F.leaky_relu(self.layer1(x)) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x
    
    

if __name__ == '__main__':
    
    model = SSRN(9,49)
    model.train()
    x=torch.randn(64,103,7,7)
    target=torch.randint(0,9,(64,))
    y=model(x)
    criterion = nn.CrossEntropyLoss()
    out=criterion(y,target)
    summary(model, (10,103,7,7),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=15,row_settings=['var_names'],depth=6)
