import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SSUN(nn.Module):
    def __init__(self, time_Step,nb_feature,num_PC,row,col,nb_classes):
        super(SSUN,self).__init__()
        self.in_ch = num_PC
        self.row = row
        self.col = col
        self.lstm = nn.LSTM(input_size = nb_feature,hidden_size = 128,batch_first = True,num_layers = 1,bias = True)
        self.dense_l1 = nn.Linear(128,128,bias=True)
        self.dense_l2 = nn.Linear(128,nb_classes,bias=True)
        self.cnn1 = nn.Conv2d(num_PC,32,3,padding=1,bias = True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.flatten1 = nn.Flatten()
        self.dense_c1 = nn.Linear(self.flatten_num(1),128)
        self.cnn2 = nn.Conv2d(32,32,3,padding=1,bias = True)
        self.pool2 = nn.MaxPool2d(2,2)
        self.flatten2 = nn.Flatten()
        self.dense_c2 = nn.Linear(self.flatten_num(2),128)
        self.cnn3 = nn.Conv2d(32,32,3,padding=1,bias = True)
        self.pool3 = nn.MaxPool2d(2,2)
        self.flatten3 = nn.Flatten()
        self.dense_c3 = nn.Linear(self.flatten_num(3),128)
        self.dense_c4 = nn.Linear(128,nb_classes)
        self.dense_join1 = nn.Linear(256,128)
        self.dense_join2 = nn.Linear(128,nb_classes)
        self.apply(self.weight_init)
    
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
        elif  isinstance(m,nn.LSTM):
            for name, param in m.named_parameters():
                nn.init.uniform_(param,-0.1,0.1)

    def flatten_num(self,flat_num):
        x = torch.zeros((1,self.in_ch,self.row,self.col))
        x = self.pool1(self.cnn1(x))
        if flat_num == 1:
            return self.flatten1(x).shape[-1]
        x = self.pool2(self.cnn2(x))
        if flat_num == 2:
            return self.flatten2(x).shape[-1]
        x = self.pool3(self.cnn3(x))
        if flat_num == 3:
            return self.flatten3(x).shape[-1]
        
    def forward(self,*x_lstm_cnn):
        ##x_lstm_cnn = (x_lstm,x_cnn)==>x_lstm (batch_size, time_step, input_dim)
        # h_0 = torch.ones((1,x_lstm_cnn[0].shape[0],128)).to(x_lstm_cnn[0].device)
        # c_0 = torch.ones((1,x_lstm_cnn[0].shape[0],128)).to(x_lstm_cnn[0].device)
        out_l,_ =self.lstm(x_lstm_cnn[0])
        # out_l = out_l[:,-1,:].contiguous()
        # out_l = out_l.view(out_l.shape[0],-1)
        LSTMDense = F.relu(self.dense_l1(out_l[:,-1,:]))
        out_lstm = self.dense_l2(LSTMDense)
        
        out_c1 = self.pool1(F.relu(self.cnn1(x_lstm_cnn[1])))
        dense_c1 = F.relu(self.dense_c1(self.flatten1(out_c1)))
        out_c2 = self.pool2(F.relu(self.cnn2(out_c1)))
        dense_c2 = F.relu(self.dense_c2(self.flatten2(out_c2)))
        out_c3 = self.pool3(F.relu(self.cnn3(out_c2)))
        dense_c3 = F.relu(self.dense_c3(self.flatten3(out_c3)))
        CNNDense = dense_c1+dense_c2+dense_c3
        out_cnn = self.dense_c4(CNNDense)
        
        JoinDense = F.relu(self.dense_join1(torch.cat([LSTMDense,CNNDense],dim = 1)))
        out_join = self.dense_join2(JoinDense)
        
        return [out_lstm,out_cnn,out_join]
    
if __name__ == '__main__':
    ## 103 200 
    net = SSUN(3,66,4,28,28,16)
    net.eval()
    summary(net, input_data=(torch.randn((1,3,66)),torch.randn((1,4,28,28))), col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=4,device='cpu')
        
        
        
        
        