# LA-DT Framework: Extended Appendices

This directory contains supplementary technical materials for the LA-DT (Look-Ahead Digital Twin) publication that are referenced in the main paper but excluded from the 22-page LLNCS format constraint.

## Contents

### Appendix A: Rigorous Proofs of Theorem 1
**File:** `appendix_a_proofs.tex`

Contains the complete formal proofs for Theorem 1 (VGR-Based Byzantine Attribution) with:
- Full statement of Theorem 1 under linear drift assumption
- Chernoff bound concentration analysis
- Empirical validation results from 1000 synthetic windows
- Discussion of failure modes and scope limitations
- References to related literature on Byzantine consensus and anomaly detection

**Key Sections:**
- A.1: Theorem 1 complete statement
- A.5: Experimental validation with VGR separation metrics
- A.6: Failure modes and limitation boundaries

### Appendix B: Comprehensive Adversarial Robustness Evaluation
**File:** `appendix_b_adversarial_evaluation.tex`

Documents LA-DT's robustness evaluation across 8 attack classes:
- **S1-S2:** Fully supported attacks (Linear & Exponential Drift)
- **S3-S5:** Partially supported attacks (Polynomial, Frogging, Natural Drift Mimicry)
- **S6-S8:** Out-of-scope attacks (FDI step-changes, majority compromise, seasonal mimicry)

**Evaluation Details:**
- Detection metrics (Precision, Recall, F1)
- Attribution accuracy per attack class
- Honeypot detection thresholds
- Recommendations for hybrid defense strategies

**Critical Finding:** Natural Drift Mimicry (S5) can evade attribution but NOT detection. Requires external reference measurements for differentiation.

---

## How to Use These Appendices

1. **For Theorem 1 Proof Details:** See `appendix_a_proofs.tex`
   - Mathematical rigor: Formal concentration bounds
   - Chernoff inequality application
   - Assumption validation under controlled conditions

2. **For Threat Model Boundary Conditions:** See `appendix_b_adversarial_evaluation.tex`
   - Complete attack classification matrix
   - Performance degradation curves
   - Defense recommendations per attack class

3. **Integration with Main Paper:**
   - Main paper references these appendices for formal justification
   - Sections 4 (Methodology) and 6 (Results) build on the theoretical frameworks here
   - Ablation studies validate the multi-metric LLR design

---

## Citation

If you reference the appendices in publications or technical reports, cite them as:

```bibtex
@inproceedings{bourreau_la-dt_2026,
  title={LA-DT: A Look-Ahead Digital Twin Framework for Proactive Byzantine Attack Attribution in IoT Sensor Networks},
  author={Bourreau, Hugo and Dagnat, Fabien and Jaafar, Fehmi and Bouchard, Kevin and Pahl, Marc-Oliver},
  booktitle={IEEE S\&P 2026},
  year={2026},
  appendix={https://github.com/SkYiMlTo/la-dt/appendix}
}
```

---

## Implementation Details

Both appendix files are in LaTeX format (.tex) and include:
- Complete mathematical notation consistent with the main paper
- Reference tables with empirical results
- Figure captions and table numbers matching the paper

For compilation or further processing, integrate these files into the main `ladt_framework.tex` using:
```latex
\appendix
\input{appendix/appendix_a_proofs.tex}
\input{appendix/appendix_b_adversarial_evaluation.tex}
```

---

**Last Updated:** February 22, 2026
**Framework Version:** 1.0 (Publication release)
