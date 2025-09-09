import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np

# --- Simple GCN Model ---
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# # --- Prepare Graph Data Object ---
# def build_data(features, labels, edge_index):
#     # Convert to torch tensors
#     x = torch.tensor(features, dtype=torch.float)
#     y = torch.tensor(labels, dtype=torch.long)
#     edge_index = torch.tensor(edge_index, dtype=torch.long)

#     # Train/val/test split
#     idx = np.arange(len(y))
#     train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
#     train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=y[train_idx], random_state=42)

#     train_mask = torch.zeros(len(y), dtype=torch.bool)
#     val_mask = torch.zeros(len(y), dtype=torch.bool)
#     test_mask = torch.zeros(len(y), dtype=torch.bool)

#     train_mask[train_idx] = True
#     val_mask[val_idx] = True
#     test_mask[test_idx] = True

#     return Data(x=x, y=y, edge_index=edge_index, 
#                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

def build_data(features, labels, edge_index):
    """
    Simple version: use all data for training (no validation/test for prototype)
    """
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = edge_index.clone().detach()

    num_nodes = len(y)
    
    # Use all data for training
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    return Data(x=x, y=y, edge_index=edge_index, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# --- Training Loop ---
def train(model, data, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# # --- Evaluation ---
# def test(model, data):
#     model.eval()
#     out = model(data.x, data.edge_index)
#     pred = out.argmax(dim=1)
#     acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
#     print(f"Test Accuracy: {acc:.4f}")
#     return acc

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred == data.y).sum() / len(data.y)  # Test on all data
    print(f"Accuracy: {acc:.4f}")
    return acc