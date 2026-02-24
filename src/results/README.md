# Arrays Directory

Numerical results in documentation and data formats.

## Contents

### LaTeX Tables (.tex)
Publication-ready LaTeX table definitions for inclusion in papers:
- `table_robustness_empirical.tex` - Byzantine attack robustness (F1 scores)
- `table_horizon_empirical.tex` - Attribution accuracy at multiple horizons
- `table_scalability_empirical.tex` - Scalability benchmarks (N=5 to 100 sensors)
- `table_realworld_empirical.tex` - Real-world validation (SWAT + AI datasets)
- `table_ablation_empirical.tex` - Signal contribution ablation study
- `table_swat_attribution_empirical.tex` - SWAT-specific attribution analysis

### Markdown Tables (.md)
Machine-readable and GitHub-friendly table formats:
- Used for documentation and README files
- Easier to review in version control than LaTeX

### JSON Results
- `experiment_results.json` - Complete raw results from all 7 experiments
- Each experiment's metrics stored for reproducibility and further analysis

## Usage

Import LaTeX tables into your paper:
```latex
\input{src/results/arrays/table_robustness_empirical.tex}
```

Convert to other formats with pandoc:
```bash
pandoc table_robustness_empirical.tex -t markdown -o table_robustness_empirical.md
```
