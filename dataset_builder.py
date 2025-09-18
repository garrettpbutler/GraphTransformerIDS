import pandas as pd
import torch
from torch_geometric.data import Data

def create_time_feature(row):
    """
    Create a normalized time feature from hour, minute, second, microsecond
    Returns a value between 0 and 1 representing the time of day
    """
    try:
        hour = float(row.get("hour", 0))
        minute = float(row.get("minute", 0))
        second = float(row.get("second", 0))
        microsecond = float(row.get("microsecond", 0))
        
        # Convert to total seconds in the day
        total_seconds = hour * 3600 + minute * 60 + second + microsecond / 1_000_000
        
        # Normalize to [0, 1] range (86400 seconds in a day)
        normalized_time = total_seconds / 86400.0
        
        return normalized_time
        
    except (ValueError, TypeError):
        return 0.0  # Default value if time data is missing

def build_graph_from_csv(df):
    if df is None:
        raise ValueError("df must be provided")

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
            protocol_feat = [1, 0]
            frame = row["No."]
            frame_length = row["Length"]
            stNum = row["StNum"]
            sqNum = row["SqNum"]
            boolean = int(row["Boolean"])
            time = create_time_feature(row)
            link_src = -1
            link_dst = -1
            fir_feat = -1
            fin_feat = -1
            seq_feat = -1
            iin_bits = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            primary_func = [-1, -1, -1, -1]
            second_func = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        elif proto == "dnp3":
            protocol_feat = [0, 1]
            frame = row["frame_number"]
            frame_length = row["frame_length"]
            stNum = -1
            sqNum = -1
            boolean = -1
            time = create_time_feature(row)
            dfc_feat = row["dfc_bit"]
            link_src = row["link_src"]
            link_dst = row["link_dst"]
            fir_feat = row["FIR"]
            fin_feat = row["FIN"]
            seq_feat = row["SEQ"]
            iin_bits = [row["iin_device_restart"], row["iin_device_trouble"], row["iin_digital_outputs_in_local"], 
                        row["iin_time_sync_required"], row["iin_class3_data_available"], row["iin_class2_data_available"], 
                        row["iin_class1_data_available"], row["iin_broadcast_msg_rx"], row["iin_configuration_corrupt"], 
                        row["iin_operation_already_executing"], row["iin_event_buffer_overflow"], 
                        row["iin_parameters_invalid_or_out_of_range"], row["iin_requested_objects_unknown"], 
                        row["iin_function_code_not_implemented"]]
            primary_func = [row["primary_func_1"], row["primary_func_4"], row["primary_func_14"], row["primary_func_15"]]
            second_func = [row["function_0"], row["function_1"], row["function_2"], row["function_3"], row["function_4"], 
                           row["function_5"], row["function_6"], row["function_7"], row["function_8"], row["function_9"], 
                           row["function_10"], row["function_11"], row["function_12"], row["function_13"], row["function_14"], 
                           row["function_129"], row["function_130"]]
        else:
            protocol_feat = [0, 0]
            frame_length = 0
            frame = -1
            frame_length = -1
            stNum = -1
            sqNum = -1
            boolean = -1
            time = "00:00:00.0000"
            dfc_feat = -1
            link_src = -1
            link_dst = -1
            fir_feat = -1
            fin_feat = -1
            seq_feat = -1
            iin_bits = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            primary_func = [-1, -1, -1, -1]
            second_func = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        edge_features.append(protocol_feat + [frame_length])

    # Build tensors
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Node features (empty for now)
    x = torch.zeros((len(node_map), 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, node_map

def combine_graphs(graphs_list, node_maps_list):
    """
    Combine multiple graphs into a single graph with unified node mapping
    Returns: combined_data, unified_node_map
    """
    # Collect all unique nodes
    all_nodes = set()
    for node_map in node_maps_list:
        all_nodes.update(node_map.keys())
    
    # Create unified node mapping
    unified_node_map = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Combine edge indices and attributes
    combined_edges_src = []
    combined_edges_dst = []
    combined_edge_attr = []
    
    for graph_data, node_map in zip(graphs_list, node_maps_list):
        # Convert edges to unified mapping
        for i in range(graph_data.edge_index.shape[1]):
            src_old_idx = graph_data.edge_index[0, i].item()
            dst_old_idx = graph_data.edge_index[1, i].item()
            
            # Find original node IDs
            inv_node_map = {v: k for k, v in node_map.items()}
            src_old_id = inv_node_map[src_old_idx]
            dst_old_id = inv_node_map[dst_old_idx]
            
            # Convert to unified indices
            src_new_idx = unified_node_map[src_old_id]
            dst_new_idx = unified_node_map[dst_old_id]
            
            combined_edges_src.append(src_new_idx)
            combined_edges_dst.append(dst_new_idx)
            
            # Add edge attributes if available
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                combined_edge_attr.append(graph_data.edge_attr[i].tolist())
    
    # Build combined tensors
    edge_index = torch.tensor([combined_edges_src, combined_edges_dst], dtype=torch.long)
    
    if combined_edge_attr:
        edge_attr = torch.tensor(combined_edge_attr, dtype=torch.float)
    else:
        edge_attr = None
    
    # Node features (will be populated later)
    x = torch.zeros((len(unified_node_map), 1), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, unified_node_map
