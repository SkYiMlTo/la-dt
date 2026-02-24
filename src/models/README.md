# Models Directory

Pre-trained and trained neural network models for Byzantine drift detection in IoT-CPS.

## Saved Models

Models are automatically saved to this directory when training completes.

## GAT Models

### `best_gat_model.pt`
Best GAT model checkpoint from training with early stopping.
- Saved during training via `GAT_Trainer._save_best_model()`
- Contains the model state_dict with best validation loss
- Loaded automatically for inference

### `gat_model_trained.pt`
Final trained GAT model after all training epochs.
- Saved by `train_gat()` in `src/training/gat_training_script.py`
- Full training checkpoint after convergence
- Can be used for transfer learning or fine-tuning

### GAT Architecture
- **Input**: Sensor time series (N×T format: N sensors, T timesteps)
- **Temporal Processing**: 1D convolutions per sensor
- **Spatial Processing**: Graph Attention Network layers
- **Outputs**:
  - Classification head: Binary (Byzantine/Natural)
  - Attribution head: Per-sensor commitment scores (0-1)

#### GAT Hyperparameters
- Hidden channels: 64
- Attention heads: 4
- Layers: 2 GAT layers
- Dropout: 0.2
- Learning rate: 0.001
- Early stopping patience: 8 epochs

## LSTM Models

### `anomaly_scorer.pkl`
Trained AnomalyScorer (LSTM-like statistical detector) with accumulated baseline statistics.
- Saves learned sensor baselines (EWMA + variance per node/sensor)
- Pickled state for easy serialization
- Can be used directly for anomaly detection without retraining

### LSTM Architecture
Statistical anomaly scorer using:
- **EWMA**: Exponential Weighted Moving Average for trend prediction
- **Z-score**: Normalized anomaly scoring
- **Per-sensor baselines**: Independent statistics for each (node, sensor) pair
- Lightweight alternative to full neural network LSTM

#### LSTM Hyperparameters
- Window size: 60 samples
- EWMA alpha: 0.3 (smoothing factor)
- Threshold: 4.0 (z-score for anomaly)
- Min std: 0.1 (prevents division by zero)

## Loading Models

### GAT Model
```python
from pathlib import Path
import torch
from src.models.gat_model import GAT_Config, GAT_Byzantine_Detector

# Load model
config = GAT_Config()
model = GAT_Byzantine_Detector(config)
model_path = Path(__file__).parent / "best_gat_model.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# Use for inference
with torch.no_grad():
    logits, attribution = model(x_data, edge_index)
```

### LSTM Model
```python
from pathlib import Path
from src.models.lstm_model import AnomalyScorer

# Load scorer
scorer_path = Path(__file__).parent / "anomaly_scorer.pkl"
scorer = AnomalyScorer.load(scorer_path)

# Use for inference
result = scorer.score(node_id=1, sensor="temperature", value=22.5)
if result and result.is_anomaly:
    print(f"Anomaly detected: z-score={result.z_score:.2f}")
```

## Saving Models

### GAT Model
Automatically saved during training via `GAT_Trainer`:
```python
trainer = GAT_Trainer(config, models_dir=Path("src/models"))
history = trainer.fit(train_loader, val_loader)
# Model saved to: src/models/best_gat_model.pt
```

### LSTM Model
Manually save after accumulating baselines:
```python
from pathlib import Path
from src.models.lstm_model import AnomalyScorer

scorer = AnomalyScorer(threshold=3.0)

# ... score many readings to build baselines ...
for node_id, sensor, value in sensor_readings:
    result = scorer.score(node_id=node_id, sensor=sensor, value=value)

# Save trained scorer
models_dir = Path("src/models")
scorer.save(models_dir / "anomaly_scorer.pkl")
```

## Training Logs

Training history is stored as dictionaries with keys:
- `train_loss` - Training loss per epoch
- `val_loss` - Validation loss per epoch
- `train_acc` - Training accuracy per epoch
- `val_acc` - Validation accuracy per epoch

## Reproducibility

All models are trained with:
- Fixed random seeds (100-104 for multi-seed experiments)
- Consistent hyperparameters across experiments
- Balanced datasets (50% normal, 50% Byzantine)
- Early stopping with patience=8 epochs

To regenerate models:
```bash
python src/experiments/main_run_all_experiments.py
```

This will train new GAT models on synthetic and real data, saving to `src/models/`.
