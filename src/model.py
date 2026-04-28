from typing import Any

import torch
from torch_geometric import torch_geometric

class GNNAutoEncoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size: int = 512
        self.latent_size: int = 3
        self.input_size: int = 3
        
        self.encoder1 = torch_geometric.nn.GCNConv(self.input_size, self.hidden_size)
        self.encoder2 = torch_geometric.nn.GCNConv(self.hidden_size,self.hidden_size)
        self.encoder3 = torch_geometric.nn.GCNConv(self.hidden_size,self.latent_size)
        self.decoder1 = torch_geometric.nn.GCNConv(self.latent_size,self.hidden_size)
        self.decoder2 = torch_geometric.nn.GCNConv(self.hidden_size,self.hidden_size)
        self.decoder3 = torch_geometric.nn.GCNConv(self.hidden_size,self.input_size)

    def forward(self, x, edge_index, edge_weight) -> torch.Tensor:
        # Encode
        x = torch.nn.functional.leaky_relu(self.encoder1(x,edge_index,edge_weight))
        x = torch.nn.functional.leaky_relu(self.encoder2(x,edge_index,edge_weight))
        x = torch.nn.functional.leaky_relu(self.encoder3(x,edge_index,edge_weight))

        # Decode
        x = torch.nn.functional.leaky_relu(self.decoder1(x,edge_index,edge_weight))
        x = torch.nn.functional.leaky_relu(self.decoder2(x,edge_index,edge_weight))
        x = self.decoder3(x,edge_index,edge_weight)
        return x
    
    def encode(self, x, edge_index, edge_weight) -> torch.Tensor:
        x = torch.nn.functional.leaky_relu(self.encoder1(x,edge_index,edge_weight))
        x = torch.nn.functional.leaky_relu(self.encoder2(x,edge_index,edge_weight))
        x = self.encoder3(x,edge_index,edge_weight)
        return x
    
    def decode(self, x, edge_index, edge_weight) -> torch.Tensor:
        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.leaky_relu(self.decoder1(x, edge_index, edge_weight))
        x = torch.nn.functional.leaky_relu(self.decoder2(x, edge_index, edge_weight))
        x = self.decoder3(x, edge_index, edge_weight)
        return x
