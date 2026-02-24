# LA-DT: Learning for Attack Detection in Digital Twins

A comprehensive framework for detecting anomalies and attacks in industrial control systems using Graph Attention Networks (GAT) and LSTM-based approaches with feature attribution analysis.

---

## Overview

This repository contains the complete implementation of the LA-DT framework with 7 comprehensive experiments evaluating attack detection across multiple scenarios, plus an additional cross-domain test on mechanical equipment:

1. **Attack Robustness** - Detection performance across 6 attack types
2. **Multi-Horizon Attribution** - Feature importance across time horizons (5-60 minutes)
3. **Scalability Analysis** - Performance with varying system sizes (5-100 nodes)
4. **SWAT Real-World Validation** - Evaluation on the SWAT dataset
5. **AI Dataset Evaluation** - Performance on AI power grid dataset
6. **Ablation Study** - Component importance analysis
7. **SWAT Attribution** - Feature attribution on real-world attacks

---

## Repository Structure

```
LA-DT/
├── src/                              # Main source code
│   ├── models/                       # Pre-trained models and model definitions
│   │   ├── gat_model.py             # Graph Attention Network
│   │   ├── lstm_model.py            # LSTM model variants
│   │   ├── best_gat_model.pt        # Pre-trained GAT weights
│   │   └── scalability_n*/          # Models for different sizes
│   ├── data/                        # Data processing and generation
│   │   ├── gat_data_generator.py    # Training data generation
│   │   └── raw/                     # Raw datasets
│   │       ├── swat/                # SWAT industrial dataset
│   │       ├── ai-data/             # AI power grid dataset
│   │       └── bearings/            # NASA bearings dataset
│   ├── experiments/                 # Experiment implementations
│   │   ├── exp_01_attack_robustness.py
│   │   ├── exp_02_multi_horizon.py
│   │   ├── exp_03_scalability.py
│   │   ├── exp_04_swat_validation.py
│   │   ├── exp_05_ai_dataset.py
│   │   ├── exp_06_ablation.py
│   │   ├── exp_07_swat_attribution.py
│   │   └── main_run_all_experiments.py   # Master orchestrator
│   ├── training/                    # Training utilities
│   │   ├── gat_training.py
│   │   └── lstm_training.py
│   ├── attribution/                 # Feature attribution pipeline
│   │   └── attribution_pipeline.py
│   ├── utils/                       # Utility functions
│   │   ├── utilities.py
│   │   ├── attack_data_generator.py
│   │   └── table_generator_*.py     # Result table generation (LaTeX/MD/CSV)
│   └── results/                     # Generated results and tables
│       ├── experiment_results.json
│       └── table_*.{tex,md,csv}     # Publication-ready tables
├── appendix/                       # Paper appendices
│   ├── appendix_a_proofs.tex
│   ├── appendix_b_adversarial_evaluation.tex
│   └── appendix_c_implementation.tex
└── requirements.txt                # Python dependencies

```

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- 4GB free disk space
- ~30 minutes for full experiment suite (varies by hardware)

### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd LA-DT

# Create virtual environment
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Or on Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import numpy as np; print(f'✓ PyTorch {torch.__version__}, NumPy {np.__version__}')"
```

---

## Running Experiments

### Run All Experiments (1-7)

```bash
python -m src.experiments.main_run_all_experiments
```

This will execute all 7 experiments sequentially and generate results tables in multiple formats.

### Run Specific Experiments

Run selected experiments by providing their numbers:

```bash
# Run only experiment 1 (Attack Robustness)
python -m src.experiments.main_run_all_experiments 1

# Run experiments 2, 5, and 6
python -m src.experiments.main_run_all_experiments 2 5 6

# Run experiments 3 and 4
python -m src.experiments.main_run_all_experiments 3 4

# Run experiments 4, 5, 6, 7 (Real-world and attribution)
python -m src.experiments.main_run_all_experiments 4 5 6 7
```

### Experiment Details

| # | Name | Duration | Output |
|---|------|----------|--------|
| 1 | Attack Robustness | ~5 min | F1/Precision/Recall across 8 attack types |
| 2 | Multi-Horizon Attribution | ~10 min | Attribution accuracy at time horizons 5-60 min |
| 3 | Scalability | ~15 min | Performance with 5, 10, 20, 50, 100 node systems |
| 4 | SWAT Validation | ~10 min | Real-world industrial dataset evaluation |
| 5 | AI Dataset | ~10 min | Power grid anomaly detection |
| 6 | Ablation Study | ~15 min | Component importance analysis |
| 7 | SWAT Attribution | ~20 min | Feature-level attribution on real attacks |

---

## Output Structure

After running experiments, results are saved to `src/results/`:

```
src/results/
├── experiment_results.json              # All raw results
├── table_robustness_empirical.{tex,md,csv}
├── table_horizon_empirical.{tex,md,csv}
├── table_scalability_empirical.{tex,md,csv}
├── table_realworld_empirical.{tex,md,csv}
├── table_ablation_empirical.{tex,md,csv}
├── table_swat_attribution_empirical.{tex,md,csv}
└── plots/                               # Generated visualizations
```

**Formats Generated:**
- **LaTeX (.tex)** - For paper inclusion
- **Markdown (.md)** - For documentation
- **CSV (.csv)** - For data analysis
- **JSON** - Complete raw results

---

## Key Features

### Anomaly Detection
- **Graph Attention Networks (GAT)** for system state representation
- **LSTM** variants for temporal pattern learning
- **Multi-scale** evaluation (single-node to 100-node systems)

### Feature Attribution
- Gradient-based attribution methods
- Multi-horizon temporality analysis
- Real-world attack interpretation

### Real-World Validation
- **SWAT** (Secure Water Treatment) plant dataset (384 MB)
- **AI Dataset** (Power systems) - 640 MB
- **NASA Bearings** IMS dataset support

### Robustness Evaluation
- 8 different attack types
- Multiple random seeds (5 seeds per experiment)
- Statistical significance testing

---

## Configuration

### Environment Variables

Create a `.env` file for custom settings:

```env
CUDA_VISIBLE_DEVICES=0          # GPU device(s) to use
RANDOM_SEED=42                  # Reproducibility
NUM_WORKERS=4                   # Data loader workers
RESULTS_DIR=src/results         # Output directory
```

### Python Version

Tested on:
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

---

## Data Requirements

Download and place datasets in the `src/data/raw/` directory:

### 1. SWAT Dataset (Secure Water Treatment)
**Download:** [Kaggle - SWAT Dataset](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system)

Place in `src/data/raw/swat/`:
- `normal.csv` (384 MB)
- `attack.csv` (14 MB)

### 2. AI Dataset (Power Systems)
**Download:** [OpenEI - AI for Robust Integration of AMI and Synchrophasor Data](https://data.openei.org/submissions/8345)

Place in `src/data/raw/ai-data/`:
- `scaled_PV_data.csv`
- `scaled_load_data.csv`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bourreau_la_dt_2026,
  title={LA-DT: Learning for Attack Detection in Digital Twins},
  author={Your Name},
  year={2026},
  url={https://github.com/SkYiMlTo/la-dt}
}
```

---

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Verify dependencies are installed
pip install -r requirements.txt
```

**GPU Memory Issues:**
```bash
# Run on CPU (slower but works)
CUDA_VISIBLE_DEVICES= python -m src.experiments.main_run_all_experiments
```

**Data Not Found:**
```bash
# Verify dataset locations
ls -lh src/data/raw/swat/
ls -lh src/data/raw/ai-data/
```

### Getting Help

- Review experiment docstrings: `python -c "from src.experiments.exp_01_attack_robustness import experiment_1_attack_robustness; help(experiment_1_attack_robustness)"`
- Check `src/results/experiment_results.json` for detailed output
- Open an issue on GitHub for support

---

## About

**LA-DT: Look-Ahead Digital Twin**  
Proactive Byzantine Attack Attribution in IoT-Cyber-Physical Systems

**Author:** Hugo Bourreau  
**Email:** hugo.bourreau@imt-atlantique.fr  
**Repository:** [github.com/SkYiMlTo/la-dt](https://github.com/SkYiMlTo/la-dt)  
**Version:** 1.0.0

---

## Contact

For questions or issues:
- **GitHub Issues:** [Open an issue](https://github.com/SkYiMlTo/la-dt/issues)
- **Email:** hugo.bourreau@imt-atlantique.fr

---

**License:** [MIT](LICENSE)

