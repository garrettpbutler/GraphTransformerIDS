# gnn_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np

class AnomalyGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_protocols=3):
        super(AnomalyGNN, self).__init__()
        self.num_protocols = num_protocols
        
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        
        # Anomaly detection head - fixed input size
        self.anomaly_classifier = nn.Sequential(
            nn.Linear((hidden_channels // 2) * num_protocols, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 2)  # 2 classes: normal vs anomaly
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Handle batching - get batch indices if not provided
        if batch is None:
            # If no batch provided, assume single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Group by batch and protocol, then flatten
        batch_size = batch.max().item() + 1
        
        # Reshape: (batch_size * num_protocols, features) -> (batch_size, num_protocols * features)
        x = x.view(batch_size, self.num_protocols, -1)
        x = x.reshape(batch_size, -1)  # Flatten protocol features
        
        # Anomaly classification
        x = self.anomaly_classifier(x)
        return F.log_softmax(x, dim=1)

def create_data_loaders(graph_datasets, labels, batch_size=32):
    """
    Create data loaders for train/val/test sets
    """
    from torch_geometric.loader import DataLoader
    
    # Combine graphs and labels
    for i, data in enumerate(graph_datasets):
        data.y = torch.tensor([labels[i]], dtype=torch.long)
    
    # Create data loader
    loader = DataLoader(graph_datasets, batch_size=batch_size, shuffle=True)
    return loader

def train_anomaly_detection(model, train_loader, val_loader, optimizer, criterion, epochs=100):
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
        
        # Validation with detailed metrics
        if epoch % 10 == 0:
            model.eval()
            val_metrics = evaluate_anomaly_detection(model, val_loader)
            
            print(f"Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"Val - F1: {val_metrics['f1']:.4f}")
            print(f"Val - TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, TN: {val_metrics['tn']}, FN: {val_metrics['fn']}")
            print("-" * 50)
            
            model.train()

def evaluate_anomaly_detection(model, loader):
    model.eval()
    total_loss = 0
    criterion = nn.NLLLoss()
    
    # Confusion matrix components
    tp, fp, tn, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            
            # Calculate confusion matrix
            for i in range(len(pred)):
                if batch.y[i] == 1 and pred[i] == 1:  # True Positive
                    tp += 1
                elif batch.y[i] == 0 and pred[i] == 1:  # False Positive
                    fp += 1
                elif batch.y[i] == 0 and pred[i] == 0:  # True Negative
                    tn += 1
                elif batch.y[i] == 1 and pred[i] == 0:  # False Negative
                    fn += 1
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }
