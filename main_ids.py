# main_ids.py
import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from dataset_builder import ProtocolGraphBuilder
from gnn_classifier import AnomalyGNN, create_data_loaders, train_anomaly_detection, evaluate_anomaly_detection

def load_labels_from_attack_column(csv_path):
    """
    Load labels from the 'Attack' column in the CSV files
    """
    df = pd.read_csv(csv_path)
    
    if 'Attack' not in df.columns:
        print(f"Warning: 'Attack' column not found in {csv_path}")
        return [0] * len(df)  # Default to all normal
    
    labels = []
    for _, row in df.iterrows():
        attack_value = row['Attack']
        # Handle different possible representations of attack
        if attack_value in [1, True, 'True', 'true', '1']:
            labels.append(1)
        else:
            labels.append(0)
    
    attack_count = sum(labels)
    normal_count = len(labels) - attack_count
    print(f"Label distribution: {normal_count} normal, {attack_count} attacks")
    
    return labels

def get_combined_labels(protocol_data):
    """
    Combine labels from all protocols using majority vote
    If any protocol indicates attack, mark as attack
    """
    all_labels = {}
    
    # Load labels for each protocol
    for protocol, df in protocol_data.items():
        if not df.empty and 'Attack' in df.columns:
            labels = []
            for _, row in df.iterrows():
                attack_value = row['Attack']
                if attack_value in [1, True, 'True', 'true', '1']:
                    labels.append(1)
                else:
                    labels.append(0)
            all_labels[protocol] = labels
            print(f"{protocol}: {sum(labels)} attacks out of {len(labels)} windows")
    
    # Determine number of windows (use minimum length)
    if not all_labels:
        return []
    
    num_windows = min(len(labels) for labels in all_labels.values())
    combined_labels = []
    
    for i in range(num_windows):
        window_labels = [labels[i] for labels in all_labels.values() if i < len(labels)]
        # If any protocol indicates attack, mark as attack
        combined_labels.append(1 if any(window_labels) else 0)
    
    return combined_labels

def main():
    # Configuration
    train_dir = "./train_csv"
    val_dir = "./val_csv" 
    test_dir = "./test_csv"
    
    # Initialize graph builder
    builder = ProtocolGraphBuilder()
    
    print("Loading training data...")
    train_graphs, train_protocol_data = builder.build_graph_dataset(train_dir)
    train_labels = get_combined_labels(train_protocol_data)
    
    print("Loading validation data...")
    val_graphs, val_protocol_data = builder.build_graph_dataset(val_dir)
    val_labels = get_combined_labels(val_protocol_data)
    
    print("Loading test data...")
    test_graphs, test_protocol_data = builder.build_graph_dataset(test_dir)
    test_labels = get_combined_labels(test_protocol_data)
    
    print(f"Training: {len(train_graphs)} graphs, {sum(train_labels)} attacks")
    print(f"Validation: {len(val_graphs)} graphs, {sum(val_labels)} attacks")
    print(f"Test: {len(test_graphs)} graphs, {sum(test_labels)} attacks")
    
    # Check if we have any attack data
    if sum(train_labels) == 0:
        print("WARNING: No attack data found in training set!")
        print("The model will only learn to identify normal behavior.")
    
    # Create data loaders
    train_loader = create_data_loaders(train_graphs, train_labels, batch_size=16)
    val_loader = create_data_loaders(val_graphs, val_labels, batch_size=16)
    test_loader = create_data_loaders(test_graphs, test_labels, batch_size=16)
    
    # Initialize anomaly detection model
    sample_graph = train_graphs[0]
    in_channels = sample_graph.x.shape[1]
    
    model = AnomalyGNN(
        in_channels=in_channels,
        hidden_channels=64,
        num_protocols=3
    )
    
    # Use class weights if imbalanced
    if sum(train_labels) > 0:
        weight = torch.tensor([1.0, len(train_labels) / sum(train_labels)])
        print(f"Using class weights: {weight}")
    else:
        weight = torch.tensor([1.0, 1.0])
        print("No attacks found, using equal class weights")
    
    criterion = nn.NLLLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Starting anomaly detection training...")
    train_anomaly_detection(model, train_loader, val_loader, optimizer, criterion, epochs=100)
    
    print("Testing anomaly detection...")
    test_metrics = evaluate_anomaly_detection(model, test_loader)
    print(f"Final Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix: TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")

if __name__ == "__main__":
    main()