"""
Threshold Optimizer Orchestration
==================================

Run threshold optimizers for Experiment 2 and Experiment 7.
Generates configuration JSON files for use in respective experiments.
"""

import sys
import subprocess
from pathlib import Path


def run_optimizer(exp_name: str) -> bool:
    """
    Run a specific threshold optimizer.
    
    Args:
        exp_name: 'exp_02' or 'exp_07'
    
    Returns:
        True if successful, False otherwise
    """
    script = f"src.threshold_optimization.{exp_name}.run_optimization"
    print(f"\n{'='*80}")
    print(f"Running {exp_name.upper()} Optimizer")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", script],
            cwd=Path(__file__).parent.parent.parent,
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ {exp_name} optimizer failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error running {exp_name}: {e}")
        return False


def main():
    """Run all optimizers sequentially."""
    print("\n" + "#"*80)
    print("#  LA-DT THRESHOLD OPTIMIZATION SUITE")
    print("#"*80)
    
    results = {}
    
    # Run Exp 02 optimizer
    results['exp_02'] = run_optimizer('exp_02')
    
    # Run Exp 07 optimizer
    results['exp_07'] = run_optimizer('exp_07')
    
    # Summary
    print("\n" + "#"*80)
    print("#  SUMMARY")
    print("#"*80)
    print()
    
    for exp, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {exp.upper()}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\nAll optimizers completed successfully!")
        print("\nGenerated files:")
        print("  - src/threshold_optimization/exp_02/exp_02_threshold.json")
        print("  - src/threshold_optimization/exp_07/exp_07_thresholds.json")
    else:
        print("\n✗ Some optimizers failed. Please check the output above.")
    
    print()
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
