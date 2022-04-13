import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class RNN(nn.Module):

    def __init__(self,n_bands, n_classes):
        super(RNN, self).__init__()
        self.timestep = n_bands
        self.input_size = 1
        self.hidden_size = 32
        self.layers = 2
        self.rnn = nn.RNN(input_size = self.input_size,hidden_size = self.hidden_size,
                          num_layers = self.layers,batch_first = True)
        self.fc = nn.Linear(self.hidden_size*self.timestep,n_classes)
        
    def forward(self, x):
        x = x.reshape(x.shape[0],self.timestep,self.input_size)
        out, h = self.rnn(x,None)
        out = self.fc(F.sigmoid(out.reshape(out.shape[0],-1)))
        return out
    @staticmethod
    def init_hidden(self,batchsize):
        return torch.zeros((self.layers,batchsize,self.hidden_size))

if __name__ == '__main__':
    model = RNN(102,9)
    model.train()
    x=torch.randn(64,102,1,1)
    target=torch.randint(0,9,(64,))
    h_state=None
    y,h_state=model(x,h_state)
    h_state = h_state.detach()
    criterion = nn.CrossEntropyLoss()
    out=criterion(y,target)
    summary(model,input_data=[x,h_state],col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=15,row_settings=['var_names'],depth=3)