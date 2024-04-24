from torch import nn
import torch.optim as optim


class FNN(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, activation_func):
        super(FNN, self).__init__()
        hid_dim = [in_dim] + hid_dim
        self.linears = []
        
        for pre, cur in zip(hid_dim, hid_dim[1:]):
            self.linears.append(nn.Linear(pre, cur))
        
        self.linears = nn.ModuleList(self.linears)
        self.activation = activation_func
        self.linear_out = nn.Linear(hid_dim[-1], out_dim)

    def forward(self, X):
        for linear in self.linears:
            X = self.activation(linear(X))
            
        logits = self.linear_out(X)
        return logits
