import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchinfo import summary

class M3DCNN(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(M3DCNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3,1,1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3,2,2), stride=(3,2,2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)
        self.features_size = self._get_final_flattened_size()
        
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = M3DCNN(102,9,7)
    model.train()
    x=torch.randn(64,102,7,7)
    target=torch.randint(0,9,(64,))
    y=model(x)
    criterion = nn.CrossEntropyLoss()
    out=criterion(y,target)
    summary(model,(10,102,7,7),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=15,row_settings=['var_names'],depth=1)