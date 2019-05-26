import torch
import torch.nn as nn

class nnNet(torch.nn.Module):
    def __init__(self, n_hidden, layers_nodes, activate_func, device):
        super(nnNet, self).__init__()
        self.n_hidden = n_hidden
        self.layers_nodes = layers_nodes
        self.fc = []
        self.device = device
        if activate_func == 'sigmoid':
            self.activate_func = nn.Sigmoid()
        elif activate_func == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_func == 'tanh':
            self.activate_func = nn.Tanh()
        else:
            print("No function!!!")

        for i in range(1, self.n_hidden+1):
            self.fc.append(nn.Sequential(
                nn.Linear(layers_nodes[i-1], layers_nodes[i]).to(self.device),
                nn.BatchNorm1d(layers_nodes[i]).to(self.device),
                self.activate_func
                ))
        self.fc_out = nn.Sequential(
            nn.Linear(layers_nodes[self.n_hidden], 19).to(self.device),
            nn.BatchNorm1d(19).to(self.device),
            nn.Softmax(dim=1)
            )
    def forward(self, x):
        #print(x.shape)
        for i in range(self.n_hidden):
            x = self.fc[i](x)
        out = self.fc_out(x)
        return out
    