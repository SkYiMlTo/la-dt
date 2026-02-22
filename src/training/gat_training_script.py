"""
gat_training_script.py

Train and evaluate GAT model for Byzantine drift detection.

Usage:
    python gat_training_script.py --mode train      # Train on synthetic data
    python gat_training_script.py --mode evaluate   # Evaluate trained model
    python gat_training_script.py --mode benchmark  # Run scalability benchmark
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time
import sys
from typing import Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

# Import custom modules
from gat_model import GAT_Config, GAT_Byzantine_Detector, GAT_Trainer, GAT_Evaluator
from gat_data_generator import (
    SyntheticDataGenerator,
    create_sensor_graph_fully_connected,
    create_data_loaders,
)


def custom_collate_fn(batch):
    """Custom collate function to handle edge_index properly."""
    xs, ys, attrs, edge_indices = zip(*batch)
    x_batch = torch.stack(xs)
    y_batch = torch.stack(ys)
    attr_batch = torch.stack(attrs)
    # All edge indices are the same, so just take the first
    edge_index = edge_indices[0]
    return x_batch, y_batch, attr_batch, edge_index


def train_gat(
    num_nodes: int = 5,
    sequence_length: int = 100,
    num_samples: int = 200,
    epochs: int = 50,
    batch_size: int = 16,
):
    """
    Train GAT model on synthetic Byzantine drift data.
    """
    print("=" * 80)
    print("Training GAT Byzantine Detector")
    print("=" * 80)
    
    # Configuration
    config = GAT_Config(
        input_channels=1,
        hidden_channels=64,
        output_channels=2,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=5,
        device="cpu",
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: GAT with {config.num_layers} layers, {config.num_heads} heads")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Generate data
    print(f"\nGenerating synthetic dataset...")
    generator = SyntheticDataGenerator(
        num_nodes=num_nodes,
        sequence_length=sequence_length,
        num_samples_per_class=num_samples // 2,
        random_seed=42,
    )
    X, y, node_attrs = generator.generate_dataset()
    print(f"  Dataset: {X.shape} (samples, nodes, time)")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Train-val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    attrs_train, attrs_val = node_attrs[:split], node_attrs[split:]
    
    # Create data loaders
    print(f"\nCreating data loaders...")
    num_nodes_actual = X.shape[1]
    edge_index = create_sensor_graph_fully_connected(num_nodes_actual)
    
    from torch.utils.data import DataLoader
    from gat_data_generator import SensorGraphDataset
    
    train_dataset = SensorGraphDataset(X_train, y_train, edge_index, attrs_train)
    val_dataset = SensorGraphDataset(X_val, y_val, edge_index, attrs_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Train
    print(f"\nTraining...")
    trainer = GAT_Trainer(config)
    history = trainer.fit(train_loader, val_loader)
    
    # Save model
    model_path = Path("gat_model_trained.pt")
    torch.save(trainer.model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Print results
    print(f"\nTraining Results:")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")
    print(f"  Best val accuracy: {max(history['val_acc'])*100:.2f}%")
    print(f"  Final train accuracy: {history['train_acc'][-1]*100:.2f}%")
    print(f"  Final val accuracy: {history['val_acc'][-1]*100:.2f}%")
    
    return trainer, history


def evaluate_gat(
    model_path: str = "gat_model_trained.pt",
    num_nodes: int = 5,
    sequence_length: int = 100,
    num_test_samples: int = 100,
):
    """
    Evaluate trained GAT model.
    """
    print("=" * 80)
    print("Evaluating GAT Byzantine Detector")
    print("=" * 80)
    
    # Load model
    config = GAT_Config()
    model = GAT_Byzantine_Detector(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate test data
    print(f"\nGenerating test dataset...")
    generator = SyntheticDataGenerator(
        num_nodes=num_nodes,
        sequence_length=sequence_length,
        num_samples_per_class=num_test_samples // 2,
        random_seed=123,  # Different seed
    )
    X_test, y_test, attrs_test = generator.generate_dataset()
    
    edge_index = create_sensor_graph_fully_connected(num_nodes)
    
    # Evaluate
    print(f"\nEvaluating...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            y = torch.tensor(y_test[i:i+1], dtype=torch.long)
            
            logits, _ = model(x, edge_index)
            pred = logits.argmax(dim=1)
            
            correct += (pred == y).sum().item()
            total += 1
    
    accuracy = correct / total
    print(f"  Test accuracy: {accuracy*100:.2f}%")
    
    return accuracy


def benchmark_scalability():
    """
    Benchmark GAT vs LSTM-style complexity across different network sizes.
    """
    print("=" * 80)
    print("Scalability Benchmark: GAT vs LSTM")
    print("=" * 80)
    
    evaluator = GAT_Evaluator()
    benchmark = evaluator.benchmark_complexity([5, 10, 20, 50, 100])
    evaluator.print_benchmark_summary(benchmark)
    
    # Timing benchmark for actual model
    print("\nPractical timing benchmark (CPU-only):")
    print("N | GAT Inference (ms)")
    print("-" * 30)
    
    for n_nodes in [5, 10, 20, 50]:
        config = GAT_Config(device="cpu")
        model = GAT_Byzantine_Detector(config)
        model.eval()
        
        x = torch.randn(10, n_nodes, 100)  # 10 samples
        edge_index = create_sensor_graph_fully_connected(n_nodes)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model(x, edge_index)
            elapsed = (time.time() - start) * 100  # Convert to ms
        
        print(f"{n_nodes:2d} | {elapsed:8.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GAT Byzantine Drift Detector")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "benchmark"],
        default="train",
        help="Run mode",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-nodes", type=int, default=5, help="Number of sensor nodes")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--num-samples", type=int, default=200, help="Total samples")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        trainer, history = train_gat(
            num_nodes=args.num_nodes,
            sequence_length=args.seq_len,
            num_samples=args.num_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.mode == "evaluate":
        accuracy = evaluate_gat()
        print(f"\nEvaluation complete. Test accuracy: {accuracy*100:.2f}%")
    elif args.mode == "benchmark":
        benchmark_scalability()
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
