"""
gat_data_generator.py
=======================

Data generation utilities for GAT model training and evaluation:
- SyntheticDataGenerator: Creates baseline normal and Byzantine-attacked sensor data
- SensorGraphDataset: PyTorch Dataset for graph-structured sensor data
- Graph utilities: Create sensor connectivity graphs
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class SyntheticDataGenerator:
    """
    Generates synthetic sensor data for Byzantine drift detection.
    
    Creates balanced datasets of normal (no attack) vs Byzantine-attacked sensor readings.
    Useful for training and evaluating GAT models before deploying on real ICS data.
    """
    
    def __init__(
        self,
        num_nodes: int = 5,
        sequence_length: int = 100,
        num_samples_per_class: int = 100,
        random_seed: int = 42,
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            num_nodes: Number of sensors in the network
            sequence_length: Length of each time series window
            num_samples_per_class: Number of samples per class (normal + Byzantine)
            random_seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.num_samples_per_class = num_samples_per_class
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_natural_drift(self) -> np.ndarray:
        """
        Generate a single sample of natural sensor drift.
        
        Returns:
            Array of shape (num_nodes, sequence_length) with natural drift patterns
        """
        # Start from random baseline
        baseline = np.random.randn(self.num_nodes) * 0.5
        
        # Add slow drift (natural variation over time)
        drift_rate = np.random.randn(self.num_nodes) * 0.001
        x = np.zeros((self.num_nodes, self.sequence_length))
        
        for t in range(self.sequence_length):
            x[:, t] = baseline + drift_rate * t + np.random.randn(self.num_nodes) * 0.1
        
        return x.astype(np.float32)
    
    def generate_byzantine_drift(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample with Byzantine drift on 1-2 randomly selected nodes.
        
        Returns:
            Tuple of:
            - x: Data with Byzantine drift shape (num_nodes, sequence_length)
            - attrs: Binary attributes indicating compromised nodes (num_nodes,)
        """
        # Start with natural drift
        x = self.generate_natural_drift()
        attrs = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Select 1-2 nodes to compromise
        compromised = np.random.choice(self.num_nodes, size=np.random.randint(1, 3), replace=False)
        
        # Add Byzantine drift (linear trend or jump)
        for node in compromised:
            if np.random.rand() < 0.5:
                # Linear drift
                x[node, :] += np.linspace(0, 1, self.sequence_length) * 0.5
            else:
                # Step change
                onset = np.random.randint(20, 60)
                x[node, onset:] += 0.8
            
            attrs[node] = 1.0
        
        return x.astype(np.float32), attrs
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a balanced dataset of normal and Byzantine samples.
        
        Returns:
            Tuple of:
            - X: Data of shape (2*num_samples_per_class, num_nodes, sequence_length)
            - y: Labels of shape (2*num_samples_per_class,) - 0 for normal, 1 for Byzantine
            - node_attrs: Attributes of shape (2*num_samples_per_class, num_nodes)
        """
        X_list = []
        y_list = []
        attrs_list = []
        
        # Generate normal samples
        for _ in range(self.num_samples_per_class):
            x = self.generate_natural_drift()
            X_list.append(x)
            y_list.append(0)
            attrs_list.append(np.zeros(self.num_nodes, dtype=np.float32))
        
        # Generate Byzantine samples
        for _ in range(self.num_samples_per_class):
            x, attrs = self.generate_byzantine_drift()
            X_list.append(x)
            y_list.append(1)
            attrs_list.append(attrs)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        node_attrs = np.array(attrs_list, dtype=np.float32)
        
        # Shuffle
        idx = np.random.permutation(len(X))
        return X[idx], y[idx], node_attrs[idx]


class SensorGraphDataset(Dataset):
    """
    PyTorch Dataset for graph-structured sensor data.
    
    Each sample is a sensor reading across N nodes over T timesteps,
    paired with the sensor connectivity graph and ground truth labels.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        edge_index: torch.Tensor,
        node_attrs: np.ndarray,
    ):
        """
        Initialize sensor graph dataset.
        
        Args:
            X: Sensor data of shape (num_samples, num_nodes, sequence_length)
            y: Labels of shape (num_samples,) - 0 for normal, 1 for Byzantine
            edge_index: Graph connectivity of shape (2, num_edges)
            node_attrs: Node attributes indicating compromised sensors (num_samples, num_nodes)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.edge_index = edge_index.long()
        self.node_attrs = torch.from_numpy(node_attrs).float()
        
        assert len(X) == len(y) == len(node_attrs), "Mismatched sample counts"
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (x, y, node_attrs, edge_index)
        """
        return self.X[idx], self.y[idx], self.node_attrs[idx], self.edge_index


def create_sensor_graph_fully_connected(num_nodes: int) -> torch.Tensor:
    """
    Create a fully-connected sensor graph.
    
    Args:
        num_nodes: Number of nodes in the graph
    
    Returns:
        Edge index tensor of shape (2, num_edges) for fully-connected graph
    """
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    node_attrs: np.ndarray,
    num_nodes: int,
    batch_size: int = 32,
    train_split: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        X: Sensor data of shape (num_samples, num_nodes, sequence_length)
        y: Labels of shape (num_samples,)
        node_attrs: Node attributes of shape (num_samples, num_nodes)
        num_nodes: Number of nodes in the graph
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    np.random.seed(random_seed)
    
    # Create edge matrix
    edge_index = create_sensor_graph_fully_connected(num_nodes)
    
    # Split data
    split = int(len(X) * train_split)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    attrs_train, attrs_val = node_attrs[:split], node_attrs[split:]
    
    # Create datasets
    train_ds = SensorGraphDataset(X_train, y_train, edge_index, attrs_train)
    val_ds = SensorGraphDataset(X_val, y_val, edge_index, attrs_val)
    
    # Custom collate function for proper edge_index handling
    def collate_fn(batch):
        xs, ys, attrs, edge_indices = zip(*batch)
        return torch.stack(xs), torch.stack(ys), torch.stack(attrs), edge_indices[0]
    
    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
