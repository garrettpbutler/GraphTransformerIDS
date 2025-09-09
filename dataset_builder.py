import pandas as pd
import torch
from torch_geometric.data import Data

def build_graph_from_csv(csv_path: str = None, df: pd.DataFrame = None):
    if df is None and csv_path is not None:
        df = pd.read_csv(csv_path)
    elif df is None:
        raise ValueError("Eitehr csv_path or df must be provided")

    node_map = {}  # maps string ID -> node index
    edges_src = []
    edges_dst = []
    edge_features = []

    def get_node_id(row, is_source=True):
        """Return a unique node identifier string for source or destination."""
        proto = row["protocol"].lower()
        if proto == "goose":
            # Use MAC address
            eth_cols = [f'{"Source" if is_source else "Destination"}_eth{i}' for i in range(1, 7)]
            mac = ":".join(str(row[c]) for c in eth_cols)
            return mac
        elif proto == "dnp3":
            # Use IP:Port
            ip_cols = [f'{"source" if is_source else "destination"}_ip{i}' for i in range(1, 5)]
            ip = ".".join(str(row[c]) for c in ip_cols)
            port = row["source_port"] if is_source else row["destination_port"]
            return f"{ip}:{port}"
        else:
            return f"unknown_{'src' if is_source else 'dst'}"

    # Iterate rows and construct graph
    for _, row in df.iterrows():
        src_id = get_node_id(row, is_source=True)
        dst_id = get_node_id(row, is_source=False)

        # Assign node indices if new
        if src_id not in node_map:
            node_map[src_id] = len(node_map)
        if dst_id not in node_map:
            node_map[dst_id] = len(node_map)

        src_idx = node_map[src_id]
        dst_idx = node_map[dst_id]

        edges_src.append(src_idx)
        edges_dst.append(dst_idx)

        # Edge features
        proto = row["protocol"].lower()
        if proto == "goose":
            proto_feat = [1, 0]
        elif proto == "dnp3":
            proto_feat = [0, 1]
        else:
            proto_feat = [0, 0]

        frame_length = row.get("frame_length", 0)
        edge_features.append(proto_feat + [frame_length])

    # Build tensors
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Node features (empty for now)
    x = torch.zeros((len(node_map), 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, node_map