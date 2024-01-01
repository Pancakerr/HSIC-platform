
import torch.nn as nn
from torchinfo import summary

def BottleneckV1(in_channels, out_channels, stride):
  return  nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class MobileNetV1(nn.Module):
    def __init__(self,in_chs = 3, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_chs,out_channels=32,kernel_size=3,stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            BottleneckV1(32, 64, stride=1),
            BottleneckV1(64, 128, stride=1),
            BottleneckV1(128, 128, stride=1),
            BottleneckV1(128, 256, stride=2),
            BottleneckV1(256, 256, stride=1),
            BottleneckV1(256, 512, stride=2),
            # BottleneckV1(512, 512, stride=1),
            # BottleneckV1(512, 512, stride=1),
            # BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 512, stride=1),
            BottleneckV1(512, 1024, stride=1),
            # BottleneckV1(1024, 1024, stride=1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=1024,out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)
        # self.softmax = nn.Softmax(dim=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottleneck(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        out = self.linear(x)
        # out = self.softmax(x)
        return out

if __name__=='__main__':
    model = MobileNetV1(144,16)
    # input = torch.randn(1, 144, 224, 224)
    # out = model(input)
    summary(model, (1, 144, 25, 25),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=15,row_settings=['var_names'],depth=5)

