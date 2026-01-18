import torch.nn.functional as F
import torch.nn as nn

class mlp(nn.Module):

    def __init__(self,
        input_dim,
        layers_num,
        hidden_dim,
        output_dim,
        with_basis,
        device
    ):
        super().__init__()
        self.device = device
        layers = []
        for i in range(layers_num):
            if i == 0: 
                layers.append(nn.Linear(input_dim, hidden_dim, with_basis))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, with_basis))
        
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, output_dim, with_basis)
        self.to(self.device)

    def forward(self, data):
        for k,l in enumerate(self.layers):
            if k==0:
                h = F.relu((l(data)))
            else:
                h = F.relu(l(h))
        result = self.lout(h).squeeze(1)
        return result
    
    




