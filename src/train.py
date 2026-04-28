import numpy as np
import torch
from rich import print
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW
from model import GNNAutoEncoder

import matplotlib.pyplot as plt

auto_encoder: GNNAutoEncoder = GNNAutoEncoder()
criterion = MSELoss()
optimizer = AdamW(params=auto_encoder.parameters(), lr=1e-3)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# Test-training split
training_test_split_boundary = int(len(dataset) * .9)
training_dataset = dataset[:training_test_split_boundary]
test_dataset = dataset[training_test_split_boundary:]

loader = DataLoader(training_dataset, batch_size=64, shuffle=True) # type: ignore

print("[bold green]BEGIN TRAINING[/bold green]")
auto_encoder.train()
for epoch in range(50): # Train for 200 epochs
    print(f"Epoch {epoch}")
    for batch in loader:
        optimizer.zero_grad()
        prediction = auto_encoder(x=batch.x, edge_index=batch.edge_index, edge_weight=batch.edge_weight)

        loss: torch.Tensor = criterion(prediction, batch.x)
        loss.backward()
        optimizer.step()

print("[bold green]BEGIN VALIDATION[/bold green]")
average_error = 0
auto_encoder.eval()

all_loss = []

for graph in test_dataset: # type: ignore
    with torch.no_grad():
        print(f"Original representation: {graph.x}")
        latent = auto_encoder.encode(graph.x, graph.edge_index, graph.edge_weight)
        print(f"Latent representation: {latent}")
        decoded = auto_encoder.decode(latent, graph.edge_index, graph.edge_weight)
        print(f"Decoded representation: {decoded}")
        loss: torch.Tensor = criterion(decoded, graph.x)
        average_error += loss.item()
        all_loss.append(loss.item())

print(f"Average loss: {average_error/len(test_dataset)}")