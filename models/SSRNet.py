
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.nn.init as init

## set activate function
def get_act_fn(act_name,inplace = True):
    if act_name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU(inplace=inplace)
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'softmax':
        return nn.Softmax(dim = 1)
    elif act_name == None:
        return None

class conv2d_layer(nn.Module):
    def __init__(self, in_chs, out_chs, kn_size, dilation = 1,padding = 0, stride = 1, norm = 'BN', act = 'leaky_relu' ,group = 1,bias = False):
        super(conv2d_layer, self).__init__()
        self.conv_layer = nn.Conv2d(in_chs,out_chs,kn_size,padding=padding,stride=stride, groups=group,dilation=dilation,bias=bias)
        self.isnorm = norm
        self.act = get_act_fn(act)
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(out_chs,affine=True)
        elif norm == 'SN':
            self.norm = nn.LayerNorm(out_chs,elementwise_affine=True)
        elif norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_chs,affine=True)
    def forward(self,x):
        x = self.conv_layer(x)
        if self.isnorm != None: 
            if self.isnorm == 'SN':
                x = self.norm(x.transpose(1,3)).transpose(1,3)
            else: x = self.norm(x)
        if self.act!= None: 
            x = self.act(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=3,act = 'leaky_relu'):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            get_act_fn(act),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel, ratio = 3, att_mode = 'CBAM',act = 'leaky_relu'):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel,ratio)
        self.spatial_attention = SpatialAttentionModule()
        self.mode = att_mode
    def forward(self, x):
        if self.mode == 'CBAM' or self.mode == 'CAM':
            x = self.channel_attention(x) * x
        if self.mode == 'CBAM' or self.mode == 'SAM':
            x = self.spatial_attention(x) * x
        return x

class LocalChannelAttention(nn.Module):
    def __init__(self, in_planes, patch_size, ratio=3, act = 'leaky_relu'):
        super(LocalChannelAttention, self).__init__()
        self.patch_size = patch_size
        self.avg_pool = nn.AvgPool2d((patch_size,patch_size),stride=patch_size)
        self.max_pool = nn.MaxPool2d((patch_size,patch_size),stride=patch_size)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = get_act_fn(act)
        self.sigmoid = nn.Sigmoid()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    def forward(self, x):
        sz = x.size()[-2:]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        out = F.interpolate(out,size = sz,mode = 'nearest')
        return out*x

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, act = 'leaky_relu'):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Sequential(
            conv2d_layer(inplanes,planes,1,norm = None,act = None,bias=False),
            conv2d_layer(planes,planes,kernel_size,group=planes,padding = padding,dilation=dilation,norm = 'BN',act = act, bias=False)
            )
    def forward(self, x):
        out = self.atrous_conv(x)
        return out

class LCA_SFCN(nn.Module):
    def __init__(self, input_channels, patch_size,n_classes,ratio = 4,hid_layer =2,hid_num = 64, act = 'leaky_relu',att_mode = 'LCA'):
        super(LCA_SFCN, self).__init__()
        self.SFCN = nn.Sequential()
        if att_mode == 'LCA':
            self.SFCN.add_module('Atte',LocalChannelAttention(input_channels,ratio = ratio,patch_size=patch_size, act = act))
        elif att_mode in ['CBAM','CAM','SAM']:
            self.SFCN.add_module('Atte',CBAM(input_channels,ratio = ratio,att_mode=att_mode,act = act))
        self.SFCN.add_module('conv',conv2d_layer(input_channels,hid_num,1,norm ='BN', act = act))
        for i in range(hid_layer):
            if att_mode == 'LCA':
                self.SFCN.add_module(f'Atte_{i}',LocalChannelAttention(hid_num,ratio = ratio,patch_size=patch_size, act = act))
            elif att_mode in ['CBAM','CAM','SAM']:
                self.SFCN.add_module(f'Atte_{i}',CBAM(hid_num,ratio = ratio,att_mode=att_mode,act = act))
            self.SFCN.add_module(f'Conv1x1_{i}',conv2d_layer(hid_num,hid_num,1,norm = 'BN',act = act))
        if att_mode == 'LCA':
            self.SFCN.add_module(f'Atte_{i+1}',LocalChannelAttention(hid_num,ratio = ratio,patch_size=patch_size, act = act))
        elif att_mode in ['CBAM','CAM','SAM']:
            self.SFCN.add_module(f'Atte_{i+1}',CBAM(hid_num,ratio = ratio,att_mode=att_mode,act = act))
        self.SFCN.add_module(f'Conv1x1_{i+1}',conv2d_layer(hid_num,n_classes,1,norm = None,act = None))
    def forward(self,x):
        return self.SFCN(x)

class MS_SRM(nn.Module):
    def __init__(self, n_classes, patch_size, aspp_rate = 4, act = 'leaky_relu',BatchNorm = nn.BatchNorm2d ):
        super(MS_SRM, self).__init__()
        self.Maxpool = nn.MaxPool2d(3, 1, 1)
        dilations = [1,2,3,4]
        aspp_ch = aspp_rate*n_classes
        self.aspp1 = _ASPPModule(n_classes, aspp_ch, 3, padding=dilations[0], dilation=dilations[0], BatchNorm=BatchNorm,act = act)
        self.aspp2 = _ASPPModule(n_classes, aspp_ch, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm,act = act)
        self.aspp3 = _ASPPModule(n_classes, aspp_ch, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm,act = act)
        self.aspp4 =_ASPPModule(n_classes, aspp_ch, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm,act = act)
        
        self.proj_cov = nn.Sequential(
            conv2d_layer(4*aspp_ch,aspp_ch,1,padding =0,norm='BN',act=act,bias=False),
            conv2d_layer(aspp_ch,aspp_ch,3,padding=2,group = aspp_ch,dilation=2, norm=None,act=None,bias = False),
            conv2d_layer(aspp_ch,aspp_ch,1,padding=0,norm='BN',act=act,bias = False),
            conv2d_layer(aspp_ch,n_classes,1, norm=None,act=None,bias=False)
        )
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        out = self.Maxpool(x)
        x1 = self.aspp1(out)
        x2 = self.aspp2(out)
        x3 = self.aspp3(out)
        x4 = self.aspp4(out)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.proj_cov(self.dropout(out))
        return out


class SSRNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.constant_(m.weight, 1.)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def __init__(self, input_channels, patch_size, n_classes,ratio = 3,hid_layer =2,hid_num = 69,aspp_rate = 4, mode = 2, act = 'leaky_relu',att_mode = 'LCA'):
        super(SSRNet, self).__init__()
        self.mode = mode
        self.net = nn.Sequential()
        self.net.add_module('LCA_SFCN',LCA_SFCN(input_channels, patch_size, n_classes, ratio, hid_layer, hid_num, act,att_mode))
        for i in range(self.mode):
            self.net.add_module(f'MS_SRM_{i}',MS_SRM(n_classes, patch_size, aspp_rate))
        self.apply(self.weight_init)

    def forward(self, x):
        out = []
        for i in range(self.mode + 1):
            x = self.net[i](x)
            out.append(x)
        return out

if __name__ == '__main__':
    ## 103 200 
    net = SSRNet(103,15,n_classes=9, mode = 1,hid_layer=2, ratio = 3,hid_num=69,aspp_rate=4,att_mode='LCA')
    summary(net, input_data=torch.randn((1,103,15,15)),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=10,device='cpu')