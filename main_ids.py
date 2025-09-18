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
    
    print(f"PFL - DataFrame length: {len(df)}")
    print(f"PFL - Edge index columns: {data.edge_index.shape[1]}")
    
    if len(df) != data.edge_index.shape[1]:
        print(F"PFL - WARNING: DataFrame rows ({len(df)}) don't match edge count ({data.edge_index.shape[1]})")
    
    for idx, row in df.iterrows():
        if idx >= data.edge_index.shape[1]:
            print(f"PFL - WARNING: Index {idx} exceeds edge index bounds")
            continue
        
        proto = row["protocol"].lower()
        proto_label = 0 if proto == "goose" else 1 if proto == "dnp3" else -1
        
        if proto_label != -1:
            # Get source and destination node indices for thsi row
            src_idx = data.edge_index[0, idx].item()
            dst_idx = data.edge_index[1, idx].item()
            
            node_protocols[src_idx].append(proto_label)
            node_protocols[dst_idx].append(proto_label)
            
    # Debug: check how many nodes have protocol information
    empty_nodes = sum(1 for protocols in node_protocols.values() if not protocols)
    print(f"Nodes with no protocol info: {empty_nodes}/{len(node_protocols)}")

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
    train_csv_path = "./FusionTest_Output.csv"
    val_csv_path = "./FusionTest_Output.csv"
    test_csv_path = "./FusionTest_Output.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"DataFrame shape: {train_csv_path.shape}")
    print(f"Protocol value counts:\n{train_csv_path['protocol'].value_counts()}")
    
    train_data, train_node_map = build_graph_from_csv(train_df)
    val_data, val_node_map = build_graph_from_csv(val_df)
    test_data, test_node_map = build_graph_from_csv(test_df)

    print("Creating unified node mapping...")
    all_nodes = set(train_node_map.keys()) | set(val_node_map.keys()) | set(test_node_map.keys())
    unified_node_map = {node: idx for idx, node in enumerate(all_nodes)}
    
    print(f"Number of nodes: {len(train_node_map)}")
    print(f"Edge index shape: {train_data.edge_index.shape}")

    print(f"Total number of nodes: {len(unified_node_map)}")

    # Extract node features + labels
    print("Preparing training features and labels...")
    train_features, train_labels = prepare_features_and_labels(train_data, unified_node_map, train_df)

    print("Preparing validation features and labels...")
    val_features, val_labels = prepare_features_and_labels(val_data, unified_node_map, val_df)

    print("Preparing test features and labels...")
    test_features, test_labels = prepare_features_and_labels(test_data, unified_node_map, test_df)
    
    print(f"Number of features: {len(features)}")
    print(f"Numer of labels: {len(labels)}")
    print(f"Label distribution:\n {pd.Series(labels).value_counts()}")

    # Build PyG Data object
    train_dataset = build_data(train_features, train_labels, train_data.edge_index, split_type='train')
    val_dataset = build_data(val_features, val_labels, val_data.edge_index, split_type='val')
    test_dataset = build_data(test_features, test_labels, test_data.edge_index, split_type='test')

    # Init model
    model = SimpleGNN(in_channels=2, hidden_channels=16, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Train + test
    train(model, train_dataset, val_dataset, optimizer, criterion, epochs=50)
    test(model, test_dataset)

if __name__ == "__main__":
    main()