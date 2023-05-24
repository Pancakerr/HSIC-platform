
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import minmax_scale
a = np.array([[1,1,1,2],[2,2,2,4],[3,3,3,6]])
print(minmax_scale(a,axis = 0))
    # summary(net, input_data=torch.randn((1,200,349,1905)),col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10,row_settings=['var_names'],depth=10,device='cpu')