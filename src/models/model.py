import torch 
import torch.nn as nn 
import torch.optim as optim

class cnnModel:

    def __init__(self, model_name):
        self.model_name = model_name


class LassoModel(nn.Module):

    def __init__(self, dim_input, num_class):
        super().__init__()
        # One layer 
        self.linear = nn.Linear(dim_input, num_class, bias=True)

    def forward(self, x):
        return self.linear(x)
    

