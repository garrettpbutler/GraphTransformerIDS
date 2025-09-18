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

def build_data(features, labels, edge_index, split_type='train'):
    """
    Build Data object with appropriate mask for train/val/test
    """
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = edge_index.clone().detach()

    num_nodes = len(y)
    
    # Create masks based on split type
    if split_type == 'train':
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    elif split_type == 'val':
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.ones(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    elif split_type == 'test':
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.ones(num_nodes, dtype=torch.bool)
    else:
        raise ValueError("split_type must be 'train', 'val', or 'test'")

    return Data(x=x, y=y, edge_index=edge_index, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# --- Training Loop ---
def train(model, train_data, val_data, optimizer, criterion, epochs=50):
    model.train()
    
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x, val_data.edge_index)
                val_loss = criterion(val_out[val_data.val_mask], val_data.y[val_data.val_mask])
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred[val_data.val_mask] == val_data.y[val_data.val_mask]).sum() / val_data.val_mask.sum()
            
            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
            model.train()

# # --- Evaluation ---
# def test(model, data):
#     model.eval()
#     out = model(data.x, data.edge_index)
#     pred = out.argmax(dim=1)
#     acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
#     print(f"Test Accuracy: {acc:.4f}")
#     return acc

def test(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[test_data.test_mask] == test_data.y[test_data.test_mask]).sum() / test_data.test_mask.sum()
        print(f"Test Accuracy: {acc:.4f}")
        
        # Additional metrics
        correct = (pred[test_data.test_mask] == test_data.y[test_data.test_mask]).sum().item()
        total = test_data.test_mask.sum().item()
        print(f"Test Results: {correct}/{total} correct")
        
    return acc