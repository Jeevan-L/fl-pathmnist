"""Make results/summary.png from the CSVs in results/.
Usage: python plot.py          (real runs)
       python plot.py quick    (test files made with --quick)"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

use_quick = "quick" in sys.argv
K = 5  # number of clients

# Group CSVs by setting, e.g. fedavg_seed0 and fedavg_seed1 -> group "fedavg"
groups = {}
for f in sorted(glob.glob("results/*.csv")):
    name = os.path.basename(f)[:-4]
    if name.startswith("quick_") != use_quick:
        continue
    key = re.sub(r"_seed\d+$", "", name)
    groups.setdefault(key, []).append(pd.read_csv(f))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: global test accuracy vs round (mean +/- std over seeds)
for key, runs in groups.items():
    acc = np.stack([d["global_acc"].values for d in runs])
    rounds = runs[0]["round"].values
    ax[0].plot(rounds, acc.mean(0), label=f"{key} (n={len(runs)})")
    ax[0].fill_between(rounds, acc.mean(0) - acc.std(0), acc.mean(0) + acc.std(0), alpha=0.2)
ax[0].set(title="Global accuracy (official test set)", xlabel="Round", ylabel="Accuracy")
ax[0].legend()

# Panel 2: final-round accuracy of the global model on each client's clean local test set
width = 0.8 / len(groups)
for i, (key, runs) in enumerate(groups.items()):
    final = np.stack([[d[f"client{k}_acc"].values[-1] for k in range(K)] for d in runs])
    ax[1].bar(np.arange(K) + i * width, final.mean(0), width, yerr=final.std(0), label=key)
ax[1].set_xticks(np.arange(K) + 0.4 - width / 2)
ax[1].set_xticklabels(["0 (noisy)"] + [str(k) for k in range(1, K)])
ax[1].set(title="Final per-client accuracy", xlabel="Client", ylabel="Accuracy")
ax[1].legend()

# Panel 3: cosine similarity of each client's update to the averaged update (FedAvg if available)
key = "fedavg" if "fedavg" in groups else list(groups)[0]
for k in range(K):
    cos = np.stack([d[f"client{k}_cos_to_avg"].values for d in groups[key]]).mean(0)
    ax[2].plot(groups[key][0]["round"].values, cos, label=f"client {k}" + (" (noisy)" if k == 0 else ""))
ax[2].axhline(0, color="gray", lw=0.5)
ax[2].set(title=f"Update cosine to average ({key})", xlabel="Round", ylabel="Cosine")
ax[2].legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/summary.png", dpi=150)
print("saved results/summary.png, groups:", list(groups))