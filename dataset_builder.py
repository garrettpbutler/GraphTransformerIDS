# dataset_builder.py
import pandas as pd
import torch
import os
import glob
import ast
from torch_geometric.data import Data

class ProtocolGraphBuilder:
    def __init__(self):
        self.protocol_nodes = {
            'GOOSE': 0,
            'DNP3': 1, 
            'TCP': 2
        }
        self.num_protocols = len(self.protocol_nodes)
    
    def parse_list_features(self, feature_str):
        """Parse string representations of lists from CSV"""
        try:
            if pd.isna(feature_str):
                return []
            return ast.literal_eval(str(feature_str))
        except:
            return []
    
    def load_protocol_data(self, directory_path):
        """
        Load all protocol CSV files from a directory
        Returns: dict with protocol -> DataFrame
        """
        protocol_files = {
            'GOOSE': 'goose_windows_*.csv',
            'DNP3': 'dnp3_windows_*.csv', 
            'TCP': 'tcp_windows_*.csv'
        }
        
        protocol_data = {}
        
        for protocol, pattern in protocol_files.items():
            file_path = os.path.join(directory_path, pattern)
            matching_files = glob.glob(file_path)
            
            if matching_files:
                # Assuming one file per protocol for now
                df = pd.read_csv(matching_files[0])
                protocol_data[protocol] = df
                print(f"Loaded {protocol} data: {len(df)} windows from {matching_files[0]}")
            else:
                print(f"Warning: No {protocol} files found in {directory_path}")
                protocol_data[protocol] = pd.DataFrame()
        
        return protocol_data
    
    def create_protocol_features(self, protocol_data):
        """
        Create feature vectors for each protocol based on window data
        """
        protocol_features = {proto: [] for proto in self.protocol_nodes.keys()}
        
        for protocol, df in protocol_data.items():
            if df.empty:
                continue
                
            for _, window in df.iterrows():
                features = self._extract_protocol_features(protocol, window)
                protocol_features[protocol].append(features)
        
        return protocol_features
    
    def _extract_protocol_features(self, protocol, window_row):
        """Extract features for a specific protocol window"""
        if protocol == 'GOOSE':
            return self._extract_goose_features(window_row)
        elif protocol == 'DNP3':
            return self._extract_dnp3_features(window_row)
        elif protocol == 'TCP':
            return self._extract_tcp_features(window_row)
        else:
            return []
    
    def _extract_goose_features(self, row):
        """Extract GOOSE protocol features keeping list structures"""
        features = [
            row.get('Num_Packets', 0) / 100.0,                    # Feature 0
            row.get('Avg_Length', 0) / 1500.0,                    # Feature 1  
            float(row.get('stNum_Change', 0)),                    # Feature 2
            float(row.get('sqNum_Reset', 0)),                     # Feature 3
            float(row.get('ConfRev_Change', 0)),                  # Feature 4
            float(row.get('Boolean_Data_Change', 0)),             # Feature 5
        ]
        
        # Use the actual list lengths as features
        src_macs = self.parse_list_features(row.get('Eth_Src', '[]'))
        dst_macs = self.parse_list_features(row.get('Eth_Dst', '[]'))
        
        # Feature 6: Number of source MACs (normalized)
        features.append(len(src_macs) / 10.0)
        
        # Feature 7: Number of destination MACs (normalized)
        features.append(len(dst_macs) / 10.0)
        
        assert len(features) == 8, f"GOOSE features should be 8, got {len(features)}"
        return features
    
    def _extract_dnp3_features(self, row):
        """Extract DNP3 protocol features keeping list structures"""
        features = [
            row.get('Num_Packets', 0) / 100.0,                    # Feature 0
            row.get('Avg_DNP3_Length', 0) / 1500.0,               # Feature 1
            float(row.get('IIN_Val_Change', 0)),                  # Feature 2
            float(row.get('Data_Object_Change', 0)),              # Feature 3
        ]
        
        # Use list lengths from DNP3 data
        src_ports = self.parse_list_features(row.get('IP_Port_Src', '[]'))
        dst_ports = self.parse_list_features(row.get('IP_Port_Dst', '[]'))
        func_codes = self.parse_list_features(row.get('Function_Codes', '[]'))
        iin_bits = self.parse_list_features(row.get('IIN_Bits_Set', '[]'))
        
        # Feature 4: Number of source IP:Port pairs
        features.append(len(src_ports) / 10.0)
        
        # Feature 5: Number of destination IP:Port pairs  
        features.append(len(dst_ports) / 10.0)
        
        # Feature 6: Number of function codes used
        features.append(len([fc for fc in func_codes if fc > 0]) / 10.0)
        
        # Feature 7: Number of IIN bits set
        features.append(len(iin_bits) / 10.0)
        
        assert len(features) == 8, f"DNP3 features should be 8, got {len(features)}"
        return features

    def _extract_tcp_features(self, row):
        """Extract TCP protocol features keeping list structures"""
        features = [
            row.get('Num_Packets', 0) / 100.0,                    # Feature 0
            row.get('Avg_Length', 0) / 1500.0,                    # Feature 1
        ]
        
        # Use list lengths from TCP data
        src_ips = self.parse_list_features(row.get('IP_Port_Src', '[]'))
        dst_ips = self.parse_list_features(row.get('IP_Port_Dst', '[]'))
        tcp_flags = self.parse_list_features(row.get('TCP_Flags', '[]'))
        
        # Feature 2: Number of source IP:Port pairs
        features.append(len(src_ips) / 10.0)
        
        # Feature 3: Number of destination IP:Port pairs
        features.append(len(dst_ips) / 10.0)
        
        # Feature 4: Connection count (from your CSV)
        features.append(float(row.get('Connection_Count', 0)) / 10.0)
        
        # Features 5-7: TCP flag statistics
        if tcp_flags:
            features.append(sum(tcp_flags) / len(tcp_flags))      # Avg flag value
            features.append(max(tcp_flags))                       # Max flag value
            features.append(len([f for f in tcp_flags if f > 0]) / len(tcp_flags))  # Active flag ratio
        else:
            features.extend([0.0, 0.0, 0.0])
        
        assert len(features) == 8, f"TCP features should be 8, got {len(features)}"
        return features
    
    def build_temporal_graph(self, protocol_features, window_num):
        """
        Build graph for a specific time window
        Each protocol becomes a node, edges represent protocol interactions
        """
        # Node features: stack all protocol features for this window
        node_feature_list = []
        valid_protocols = []
        
        for protocol in self.protocol_nodes.keys():
            if (protocol in protocol_features and 
                len(protocol_features[protocol]) > window_num):
                features = protocol_features[protocol][window_num]
                node_feature_list.append(features)
                valid_protocols.append(protocol)
            else:
                # Use zero features for missing protocol data
                default_size = 8  # Standardized feature size
                node_feature_list.append([0.0] * default_size)
                valid_protocols.append(protocol)
        
        # Create fully connected graph between protocols
        edge_src, edge_dst = [], []
        
        for i, proto1 in enumerate(self.protocol_nodes.keys()):
            for j, proto2 in enumerate(self.protocol_nodes.keys()):
                if i != j:  # No self-loops
                    edge_src.append(i)
                    edge_dst.append(j)
        
        # Convert to tensors
        x = torch.tensor(node_feature_list, dtype=torch.float)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        
        # Create edge attributes (simple ones for now)
        edge_attr = torch.ones(len(edge_src), 1)  # Uniform edge weights
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   window_num=window_num, protocols=valid_protocols)
    
    def build_graph_dataset(self, directory_path):
        """
        Main function to build graph dataset from directory of CSV files
        """
        # Load protocol data
        protocol_data = self.load_protocol_data(directory_path)
        
        # Extract features
        protocol_features = self.create_protocol_features(protocol_data)
        
        # Determine number of time windows (use min to handle different lengths)
        window_counts = []
        for protocol, features in protocol_features.items():
            if features:
                window_counts.append(len(features))
        
        if not window_counts:
            raise ValueError("No protocol data found in directory")
        
        num_windows = min(window_counts)
        print(f"Building graphs for {num_windows} time windows")
        
        # Build graphs for each time window
        graph_dataset = []
        for window_num in range(num_windows):
            graph_data = self.build_temporal_graph(protocol_features, window_num)
            graph_dataset.append(graph_data)
        
        return graph_dataset
    