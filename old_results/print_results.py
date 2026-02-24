import json

with open("LA-DT/results/experiment_results.json") as f:
    data = json.load(f)

print("=== EXPERIMENT 1: 8-Attack Robustness ===")
rob = data["experiment_1_robustness"]
for name, vals in rob.items():
    print(f"  {name:30s}: F1={vals['f1']:.3f}  Prec={vals['precision']:.3f}  Rec={vals['recall']:.3f}  Acc={vals['accuracy']:.3f}")

print()
print("=== EXPERIMENT 2: Multi-Horizon Attribution ===")
hor = data["experiment_2_horizon"]
for h, vals in hor.items():
    print(f"  {h:10s}: Accuracy={vals['accuracy_pct']:.1f}%  VGR={vals['avg_vgr']:.3f}  SCD={vals['avg_scd']:.3f}  LLR={vals['avg_llr']:.3f}")

print()
print("=== EXPERIMENT 3: Scalability ===")
scl = data["experiment_3_scalability"]
for n, vals in scl.items():
    print(f"  N={n:5s}: F1={vals['f1']:.3f}  Acc={vals['accuracy']:.3f}  Inf={vals['inference_ms']:.1f}ms")

print()
print("=== EXPERIMENT 4: SWAT ===")
sw = data["experiment_4_swat"]
print(f"  F1={sw['f1']:.3f}  Acc={sw['accuracy']:.3f}  Samples={sw['total_samples']}")

print()
print("=== EXPERIMENT 5: AI Dataset ===")
ai = data["experiment_5_ai"]
print(f"  F1={ai['f1']:.3f}  Acc={ai['accuracy']:.3f}  Samples={ai['total_samples']}")

print()
print("=== EXPERIMENT 6: Ablation ===")
abl = data["experiment_6_ablation"]
for name, vals in abl.items():
    print(f"  {name:25s}: {vals['accuracy_pct']:.1f}%  Impact={vals['impact_pct']:+.1f}%")

print()
print("=== EXPERIMENT 7: SWAT Attribution ===")
sa = data["experiment_7_swat_attribution"]
if "status" not in sa:
    for h, vals in sa.items():
        print(f"  {h:5s} min: Acc={vals['accuracy_pct']:.1f}%  VGR={vals['avg_vgr']:.3f}  SCD={vals['avg_scd']:.3f}  LLR={vals['avg_llr']:.3f}")
else:
    print(f"  Status: {sa['status']}")
