import matplotlib
# Use non-interactive backend - CRITICAL FIX
matplotlib.use('Agg')  # This must be before importing pyplot
import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
from torch_geometric.utils import to_networkx

class GraphVisualizer:
    def __init__(self):
        self.protocol_colors = {
            'GOOSE': 'red',
            'DNP3': 'blue', 
            'TCP': 'green'
        }
    
    def plot_feature_evolution(self, graph_dataset, feature_names, num_windows=10):
        """
        Plot how features evolve over time windows
        """
        if num_windows > len(graph_dataset):
            num_windows = len(graph_dataset)
        
        print(f"Creating feature evolution plot for {num_windows} windows...")
        
        # Select first few windows to visualize
        selected_windows = graph_dataset[:num_windows]
        
        # Extract features for each protocol over time
        protocols = ['GOOSE', 'DNP3', 'TCP']
        time_points = range(num_windows)
        
        # Create subplots for each feature
        num_features = len(feature_names)
        fig, axes = plt.subplots(num_features, 3, figsize=(15, 4*num_features))
        
        if num_features == 1:
            axes = axes.reshape(1, -1)
        
        for feature_idx, feature_name in enumerate(feature_names):
            for protocol_idx, protocol in enumerate(protocols):
                ax = axes[feature_idx, protocol_idx]
                
                # Extract feature values for this protocol over time
                feature_values = []
                for window_data in selected_windows:
                    protocol_features = window_data.x[protocol_idx]
                    if feature_idx < len(protocol_features):
                        feature_values.append(protocol_features[feature_idx].item())
                    else:
                        feature_values.append(0.0)
                
                ax.plot(time_points, feature_values, 
                       color=self.protocol_colors[protocol], 
                       marker='o', linewidth=2)
                ax.set_title(f'{protocol} - {feature_name}')
                ax.set_xlabel('Time Window')
                ax.set_ylabel('Feature Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_evolution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved feature_evolution.png")
        plt.close()  # Important: close the figure to free memory
    
    def plot_single_window_graph(self, graph_data, window_num):
        """
        Plot the graph structure for a single time window
        """
        try:
            print(f"Creating graph visualization for window {window_num}...")
            
            # Convert to networkx graph
            G = to_networkx(graph_data, to_undirected=True)
            
            plt.figure(figsize=(10, 8))
            
            # Define node positions in a triangle layout
            pos = {
                0: (0, 1),   # GOOSE - top
                1: (-1, 0),  # DNP3 - left
                2: (1, 0)    # TCP - right
            }
            
            # Node colors
            node_colors = [self.protocol_colors['GOOSE'], 
                          self.protocol_colors['DNP3'], 
                          self.protocol_colors['TCP']]
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=2000, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            
            # Add edge labels (weights or features)
            edge_labels = {}
            for i, (src, dst) in enumerate(zip(graph_data.edge_index[0], graph_data.edge_index[1])):
                edge_labels[(src.item(), dst.item())] = f'{i+1}'
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title(f'Protocol Interaction Graph - Window {window_num}')
            plt.axis('off')
            plt.savefig(f'window_{window_num}_graph.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved window_{window_num}_graph.png")
            plt.close()  # Important: close the figure
            
        except Exception as e:
            print(f"Error plotting graph for window {window_num}: {e}")
            # Fallback: just print the graph info
            print(f"Window {window_num}: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    def plot_anomaly_detection_timeline(self, graph_dataset, predictions, labels, num_windows=50):
        """
        Plot anomaly detection results over time
        """
        if num_windows > len(graph_dataset):
            num_windows = len(graph_dataset)
        
        print(f"Creating anomaly timeline plot for {num_windows} windows...")
        
        time_points = range(num_windows)
        
        # Extract a key feature from each protocol to show patterns
        goose_features = [g.x[0][0].item() for g in graph_dataset[:num_windows]]  # GOOSE feature 0
        dnp3_features = [g.x[1][1].item() for g in graph_dataset[:num_windows]]   # DNP3 feature 1  
        tcp_features = [g.x[2][2].item() for g in graph_dataset[:num_windows]]    # TCP feature 2
        
        # Normalize features for better visualization
        def normalize(values):
            if max(values) - min(values) > 0:
                return [(v - min(values)) / (max(values) - min(values)) for v in values]
            return values
        
        goose_norm = normalize(goose_features)
        dnp3_norm = normalize(dnp3_features)
        tcp_norm = normalize(tcp_features)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Feature evolution
        ax1.plot(time_points, goose_norm, color='red', label='GOOSE', linewidth=2)
        ax1.plot(time_points, dnp3_norm, color='blue', label='DNP3', linewidth=2)
        ax1.plot(time_points, tcp_norm, color='green', label='TCP', linewidth=2)
        
        # Highlight anomaly windows
        for i in range(min(num_windows, len(labels))):
            if labels[i] == 1:  # Actual anomaly
                ax1.axvline(x=i, color='red', alpha=0.3, linestyle='--')
            if i < len(predictions) and predictions[i] == 1:  # Predicted anomaly
                ax1.axvline(x=i, color='orange', alpha=0.5, linewidth=3)
        
        ax1.set_title('Protocol Feature Evolution with Anomaly Detection')
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Normalized Feature Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores/confidence
        if len(predictions) >= num_windows:
            ax2.bar(time_points, predictions[:num_windows], 
                   color=['green' if p == 0 else 'red' for p in predictions[:num_windows]],
                   alpha=0.7)
            ax2.set_title('Anomaly Predictions (0=Normal, 1=Attack)')
            ax2.set_xlabel('Time Window')
            ax2.set_ylabel('Prediction')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('anomaly_timeline.png', dpi=300, bbox_inches='tight')
        print("✓ Saved anomaly_timeline.png")
        plt.close()
    
    def plot_protocol_correlation_heatmap(self, graph_dataset, num_windows=20):
        """
        Plot correlation heatmap between protocol features
        """
        if num_windows > len(graph_dataset):
            num_windows = len(graph_dataset)
        
        print(f"Creating correlation heatmap for {num_windows} windows...")
        
        # Extract all features for correlation analysis
        all_features = []
        for window_data in graph_dataset[:num_windows]:
            # Flatten all protocol features for this window
            window_features = window_data.x.flatten().detach().numpy()
            all_features.append(window_features)
        
        all_features = np.array(all_features)
        
        # Calculate correlation matrix (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_matrix = np.corrcoef(all_features.T)
            # Replace NaN and Inf with 0
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Create labels for the features
        protocol_names = ['GOOSE'] * 8 + ['DNP3'] * 8 + ['TCP'] * 8
        feature_names = [f'F{i}' for i in range(8)] * 3
        labels = [f'{proto}\n{feat}' for proto, feat in zip(protocol_names, feature_names)]
        
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.colorbar(im, label='Correlation Coefficient')
        plt.title('Protocol Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('protocol_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved protocol_correlation.png")
        plt.close()

    def quick_visualization(self, graph_dataset, labels=None, num_windows=30):
        """Quick plot of feature evolution"""
        if num_windows > len(graph_dataset):
            num_windows = len(graph_dataset)
        
        print(f"Creating quick visualization for {num_windows} windows...")
        
        time_points = range(num_windows)
        
        # Extract first feature from each protocol
        goose_f0 = [g.x[0][0].item() for g in graph_dataset[:num_windows]]
        dnp3_f0 = [g.x[1][0].item() for g in graph_dataset[:num_windows]]
        tcp_f0 = [g.x[2][0].item() for g in graph_dataset[:num_windows]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, goose_f0, 'r-', label='GOOSE Packets', linewidth=2)
        plt.plot(time_points, dnp3_f0, 'b-', label='DNP3 Packets', linewidth=2) 
        plt.plot(time_points, tcp_f0, 'g-', label='TCP Packets', linewidth=2)
        
        # Mark anomalies if labels provided
        if labels is not None:
            for i in range(min(num_windows, len(labels))):
                if labels[i] == 1:
                    plt.axvline(x=i, color='red', alpha=0.3, linestyle='--', label='Anomaly' if i == 0 else "")
        
        plt.title('Protocol Traffic Over Time Windows')
        plt.xlabel('Time Window (5-second intervals)')
        plt.ylabel('Normalized Packet Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('quick_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Saved quick_visualization.png")
        plt.close()