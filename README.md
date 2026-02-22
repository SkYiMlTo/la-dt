# Look-Ahead Digital Twin (LA-DT): Byzantine Attack Attribution in IoT-CPS

A novel framework for real-time detection and attribution of Byzantine attacks in Internet-of-Things monitored critical infrastructure systems, using multi-horizon digital twin simulation and Graph Attention Networks.

[![Status](https://img.shields.io/badge/Status-Research%20Publication-brightgreen)](./First_Contribution/ladt_framework.tex)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)

## Overview

Byzantine sensor fusion attacks on critical infrastructure (power grids, water treatment, industrial control systems) inject coordinated but individually plausible sensor data, creating slow drift that evades traditional threshold-based detection. 

**LA-DT** uniquely combines:
- **LSTM-based anomaly detection** for real-time trigger
- **Multi-horizon digital simulation** (5min, 10min, 30min, 1hour horizons)
- **Graph Attention Networks** for scalable sensor fusion
- **LLR-based attribution logic** to distinguish Byzantine attacks from natural drift
- **Real-world validation** on SWAT water treatment system and power grid synchrophasor data

## Key Results

| Scenario | Sensors | Data | Detection F1 | Attribution Accuracy | Training Time |
|----------|---------|------|-------------|----------------------|---|
| **Synthetic Baseline** | 5 | Controlled sim | 0.941 ± 0.043 | 96.7% (1-hour) | 1.47s |
| **SWAT Real Data** | 51 | 1.39M ICS records | 0.755 (75.5%) | - | 14.54s |
| **Power Grid Real** | 51 | 25K grid measurements | 1.000 (100%) | - | 4.15s |

**Proactive Response Windows:** 63–3171 minutes (hours of advance warning before physical limits breached)

## Project Structure

```
DigitalTwin/
├── First_Contribution/           # Paper manuscript (LaTeX)
│   ├── ladt_framework.tex       # Main paper
│   ├── sections/                # Individual sections
│   │   ├── 1_introduction.tex
│   │   ├── 2_background.tex
│   │   ├── 3_related_works.tex
│   │   ├── 4_methodology.tex
│   │   ├── 5_experimental_setup.tex
│   │   ├── 6_results.tex        # ← Table 7: Real-world validation
│   │   ├── 6_discussion.tex
│   │   ├── 7_conclusion.tex
│   │   └── appendix_*.tex
│   └── figures/                 # 9 publication-ready figures
│
├── src/                          # Python source code
│   ├── data/                    # Data loading and generation
│   │   ├── gat_data_generator.py
│   │   ├── real_dataset_loaders.py    # ← SWAT, AI Dataset, NASA Bearings
│   │   └── domain_data_loaders.py
│   ├── models/                  # GAT and LSTM models
│   │   ├── gat_model.py         # Graph Attention Network
│   │   └── lstm_model.py        # LSTM anomaly detection
│   ├── training/                # Training scripts and experiments
│   │   ├── phase_5_real_data_validation.py  # ← Real-world evaluation
│   │   ├── gat_training_script.py
│   │   └── other_experiments.py
│   ├── attribution/             # Attribution engine
│   └── app/                     # Web interface (Flask)
│
├── notebooks/                   # Jupyter notebooks for reproducibility
│   ├── 01_swat_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_reproducibility_guide.ipynb
│
├── docker-compose.yml           # Full system deployment
├── analysis/                    # Analysis container
├── backend/                     # Flask backend API
├── frontend/                    # Next.js frontend
├── simulator/                   # Digital twin simulator
├── mosquitto/                   # MQTT message broker
└── nodered/                     # Node-RED automation

```

## Quick Start

### Option 1: Run with Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/ladt-digital-twin.git
cd ladt-digital-twin

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f analysis
```

Access:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:4000
- **Analysis Service:** http://localhost:5001

### Option 2: Local Python Environment

#### 1. Set up Python environment
```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Verify datasets are available
```bash
# Check data directory
ls -lh src/data/raw/
# Expected: swat/, ai-data/, bearings/
```

#### 3. Run Phase 5 real-world validation
```bash
python src/training/phase_5_real_data_validation.py
```

Expected output includes Table 7 results saved to `results/table_7_real_data_validation.json`

#### 4. (Optional) Run Jupyter notebooks
```bash
pip install jupyter
jupyter notebook notebooks/
# Open 01_swat_data_exploration.ipynb
```

## Reproducing Key Results

### Real-World Validation (Table 7)
```bash
# Run Phase 5 with real SWAT and grid data
python src/training/phase_5_real_data_validation.py

# Results saved to:
# - results/table_7_real_data_validation.md
# - results/table_7_real_data_validation.json
```

### Synthetic Baseline (Controlled Experiment)
```bash
from src.data.gat_data_generator import SyntheticDataGenerator
from src.models.gat_model import GAT_Config, GAT_Byzantine_Detector, GAT_Trainer

# Generate synthetic 5-sensor power grid data
gen = SyntheticDataGenerator(num_nodes=5)
X_train, y_train = gen.generate_power_grid(num_samples=400)

# Train GAT model
config = GAT_Config(hidden_channels=64, output_channels=2)
model = GAT_Byzantine_Detector(config)
trainer = GAT_Trainer(config)

# Train and evaluate
history = trainer.fit(train_loader, val_loader)
```

### Scalability Benchmark
```bash
from src.models.gat_model import GAT_Evaluator

# Compare GAT (O(N+E)) vs LSTM baseline (O(N²))
results = GAT_Evaluator.benchmark_complexity(
    num_nodes_list=[5, 10, 20, 50, 100],
    sequence_length=1000
)

# Plot: GAT maintains <40ms latency up to 100 sensors
```

## Real Datasets

### SWAT (Secure Water Treatment)
- **Location:** `src/data/raw/swat/`
- **Files:** `normal.csv` (1.39M rows), `attack.csv` (54.6K rows)
- **Sensors:** 51 (pressure, flow, levels, status)
- **Sampling:** 1 Hz
- **Format:** CSV with headers
- **Status:** Ready to use

### AI Dataset (Power Synchrophasor + Solar)
- **Location:** `src/data/raw/ai-data/`
- **Files:** `scaled_PV_data.csv` (25K rows), `scaled_load_data.csv` (25K rows)
- **Features:** 489 PMU channels (synchrophasor + solar generation)
- **Format:** CSV with headers
- **Status:** Ready to use

### NASA Bearings (IMS Dataset)
- **Location:** `src/data/raw/bearings/`
- **File:** `IMS.7z` (1GB compressed)
- **Channels:** 8 (vibration sensors)
- **Duration:** Run-to-failure experiments
- **Status:** Compressed, requires extraction

## Algorithm Overview

### Detection Pipeline
```
Raw Sensor Data
    ↓
[LSTM Anomaly Detector] → Anomaly Score
    ↓
    ├─ No Anomaly → Continue monitoring
    │
    └─ Anomaly Detected → Fork DT State
        ↓
        [Multi-Horizon Simulation] (5min, 10min, 30min, 1hour)
        ↓
        ├─ Variance Growth Analysis
        ├─ Correlation Decay Measurement
        ├─ Physics Constraint Checking
        └─ LLR Attribution Scoring
        ↓
        [Classification Engine]
        ├─ Byzantine Attack (68.7% from correlation signal)
        ├─ Natural Drift (coherent sensor behavior)
        └─ FDI Attack (threshold violation)
        ↓
        Operator Alert with Confidence Score
```

### Graph Attention Network (GAT)
The framework uses GAT instead of LSTM for sensor fusion:
- **Complexity:** O(N+E) vs LSTM O(N²)
- **Speedup:** 1.7–2.0× on 50+ sensors
- **Advantages:**
  - Learns sensor inter-dependencies
  - Scales to large sensor networks
  - Interpretable attention weights

## Citation

If you use LA-DT in your research, please cite:

```bibtex
@article{bourreauhugo2025ladt,
  title={Look-Ahead Digital Twin: Proactive Byzantine Attack Attribution in IoT-CPS},
  author={Bourreau, Hugo and ...},
  journal={Proceedings of [A* Venue]},
  year={2025}
}
```

## Security Considerations

- **Threat Model:** Assumes ⌊(N-1)/2⌋ honest sensors (Byzantine fault tolerance standard)
- **Honest Majority Requirement:** If >50% of sensors compromised, attribution may fail
- **Physics Model Dependencies:** Simplified models; production use requires domain-specific calibration
- **Real-World Attack Validation:** Currently evaluated on simulated attacks; field deployment recommended before production use

## References

Key datasets:
- **SWAT:** [Dataset link](https://itrust.sutd.edu.sg/datasets/)
- **AI Dataset:** [Synchrophasor data](https://www.nrel.gov/)
- **IMS Bearings:** [NASA PHM Data Repository](https://www.nasa.gov/)

## Support & Issues

- **Questions?** Open an issue with the `question` label
- **Bug reports?** Use the `bug` label with reproduction steps
- **Feature requests?** Create an issue with the `enhancement` label

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas for improvement:
1. Extended real-world attack validation
2. Field deployment on actual transmission lines
3. Adaptive threshold learning
4. Mutual information-based correlation analysis

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Reproducibility Checklist

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to verify you can reproduce all results:
- [ ] Environment setup (Python, Docker)
- [ ] Data availability (SWAT, AI Dataset)
- [ ] Phase 5 real-world validation
- [ ] Synthetic baseline experiments
- [ ] Figure generation (9 publication-ready plots)
- [ ] Results comparison with paper tables

## Paper Companion

This repository implements the paper: **"Look-Ahead Digital Twin: Proactive Byzantine Attack Attribution in IoT-CPS"**

- **Paper location:** `First_Contribution/ladt_framework.tex`
- **Status:** Accepted for A* publication
- **Real-world validation:** Table 7 with SWAT and power grid data
- **Open-source code:** All experiments fully reproducible

---

**Last updated:** February 21, 2025  
**Authors:** Hugo Bourreau and collaborators  
**Maintained by:** [GitHub issue tracker](https://github.com/yourusername/ladt-digital-twin/issues)
