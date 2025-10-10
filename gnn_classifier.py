# gnn_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np

class TemporalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_protocols=3):
        super(TemporalGNN, self).__init__()
        self.num_protocols = num_protocols
        
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * num_protocols, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Reshape for temporal processing (if needed)
        # For now, flatten all node features for classification
        batch_size = x.shape[0] // self.num_protocols
        x = x.view(batch_size, -1)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def create_data_loaders(graph_datasets, labels, batch_size=32):
    """
    Create data loaders for train/val/test sets
    graph_datasets: list of Data objects for each time window
    labels: corresponding labels for each time window
    """
    from torch_geometric.loader import DataLoader
    
    # Combine graphs and labels
    for i, data in enumerate(graph_datasets):
        data.y = torch.tensor([labels[i]], dtype=torch.long)
    
    # Create data loader
    loader = DataLoader(graph_datasets, batch_size=batch_size, shuffle=True)
    return loader

def train_temporal(model, train_loader, val_loader, optimizer, criterion, epochs=50):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch.x, batch.edge_index, batch.batch)
                    val_loss += criterion(out, batch.y).item()
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            val_acc = correct / total if total > 0 else 0
            print(f"Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
            model.train()

def test_temporal(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    acc = correct / total if total > 0 else 0
    print(f"Test Accuracy: {acc:.4f} ({correct}/{total} correct)")
    return acc
