# main_ids.py
import pandas as pd
import torch
import torch.nn as nn
import os
from dataset_builder import ProtocolGraphBuilder
from gnn_classifier import TemporalGNN, create_data_loaders, train_temporal, test_temporal

def load_labels(csv_path):
    """
    Load labels for the time windows
    This function needs to be implemented based on your labeling scheme
    For now, returns dummy labels (all normal)
    """
    # TODO: Implement based on your actual labeling
    df = pd.read_csv(csv_path)
    return [0] * len(df)  # 0 = normal, 1 = attack

def main():
    # Configuration
    train_dir = "./train_csv"
    val_dir = "./val_csv" 
    test_dir = "./test_csv"
    
    # Initialize graph builder
    builder = ProtocolGraphBuilder()
    
    print("Loading training data...")
    train_graphs = builder.build_graph_dataset(train_dir)
    train_labels = load_labels(os.path.join(train_dir, "goose_windows_train.csv"))  # Use any protocol file
    
    print("Loading validation data...")
    val_graphs = builder.build_graph_dataset(val_dir)
    val_labels = load_labels(os.path.join(val_dir, "goose_windows_val.csv"))
    
    print("Loading test data...")
    test_graphs = builder.build_graph_dataset(test_dir)
    test_labels = load_labels(os.path.join(test_dir, "goose_windows_test.csv"))
    
    print(f"Training: {len(train_graphs)} graphs")
    print(f"Validation: {len(val_graphs)} graphs") 
    print(f"Test: {len(test_graphs)} graphs")
    
    # Create data loaders
    train_loader = create_data_loaders(train_graphs, train_labels, batch_size=16)
    val_loader = create_data_loaders(val_graphs, val_labels, batch_size=16)
    test_loader = create_data_loaders(test_graphs, test_labels, batch_size=16)
    
    # Determine input feature size from first graph
    sample_graph = train_graphs[0]
    in_channels = sample_graph.x.shape[1]
    print(f"Input feature size: {in_channels}")
    
    # Initialize model
    model = TemporalGNN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2,  # binary classification: normal vs attack
        num_protocols=3
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    # Train and test
    print("Starting training...")
    train_temporal(model, train_loader, val_loader, optimizer, criterion, epochs=100)
    
    print("Testing...")
    test_temporal(model, test_loader)

if __name__ == "__main__":
    main()