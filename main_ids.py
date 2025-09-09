# main.py
import pandas as pd
import torch
import torch.nn as nn
from dataset_builder import build_graph_from_csv
from gnn_classifier import SimpleGNN, build_data, train, test

def prepare_features_and_labels(data, node_map, df):
    """
    Build node features + labels for protocol classification.
    - One-hot protocol feature for each node
    - Label = 0 (GOOSE) or 1 (DNP3)
    """
    # # Reverse map to get index -> ID
    inv_node_map = {v: k for k, v in node_map.items()}

    # # Track protocols per node
    # node_protocols = {i: [] for i in inv_node_map.keys()}

    # for _, row in df.iterrows():
    #     proto = row["protocol"].lower()
    #     proto_label = 0 if proto == "goose" else 1 if proto == "dnp3" else -1

    #     src_id = inv_node_map.get(data.edge_index[0, _].item(), None)
    #     dst_id = inv_node_map.get(data.edge_index[1, _].item(), None)

    #     if proto_label != -1:
    #         if src_id is not None:
    #             node_protocols[src_id].append(proto_label)
    #         if dst_id is not None:
    #             node_protocols[dst_id].append(proto_label)
    
    node_protocols = {i: [] for i in range(len(node_map))}
    
    for idx, row in df.iterrows():
        proto = row["protocol"].lower()
        proto_label = 0 if proto == "goose" else 1 if proto == "dnp3" else -1
        
        if proto_label != -1:
            # Get source and destination node indices for thsi row
            src_idx = data.edge_index[0, idx].item()
            dst_idx = data.edge_index[1, idx].item()
            
            node_protocols[src_idx].append(proto_label)
            node_protocols[dst_idx].append(proto_label)

    features, labels = [], []
    for node_idx in range(len(inv_node_map)):
        if node_protocols[node_idx]:
            # Majority vote
            label = max(set(node_protocols[node_idx]), key=node_protocols[node_idx].count)
        else:
            label = 0  # default to GOOSE

        labels.append(label)
        # One-hot feature (same as label but vectorized)
        if label == 0:
            features.append([1, 0])
        else:
            features.append([0, 1])

    return features, labels

def main():
    # Load CSV into graph
    csv_path = "./FusionTest_Output.csv"
    df = pd.read_csv(csv_path)
    
    data, node_map = build_graph_from_csv(df)

    # Extract node features + labels
    features, labels = prepare_features_and_labels(data, node_map, df)

    # Build PyG Data object
    dataset = build_data(features, labels, data.edge_index)

    # Init model
    model = SimpleGNN(in_channels=2, hidden_channels=16, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Train + test
    train(model, dataset, optimizer, criterion, epochs=50)
    test(model, dataset)

if __name__ == "__main__":
    main()