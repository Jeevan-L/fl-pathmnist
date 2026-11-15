"""Make results/summary.png (main runs) and results/noise_control.png (noise vs no-noise control).
Also prints a summary table. Usage: python plot.py   (or: python plot.py quick)"""
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

control = {k: v for k, v in groups.items() if "_noise" in k}   # runs with a different noise rate
main = {k: v for k, v in groups.items() if "_noise" not in k}  # the 40%-noise runs

# ---------- Figure 1: main comparison ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel 1: global test accuracy vs round (mean +/- std over seeds)
for key, runs in main.items():
    acc = np.stack([d["global_acc"].values for d in runs])
    rounds = runs[0]["round"].values
    ax[0].plot(rounds, acc.mean(0), label=f"{key} (n={len(runs)})")
    ax[0].fill_between(rounds, acc.mean(0) - acc.std(0), acc.mean(0) + acc.std(0), alpha=0.2)
ax[0].set(title="Global accuracy (official test set)", xlabel="Round", ylabel="Accuracy")
ax[0].legend()

# Panel 2: final-round accuracy of the global model on each client's clean local test set
width = 0.8 / len(main)
for i, (key, runs) in enumerate(main.items()):
    final = np.stack([[d[f"client{k}_acc"].values[-1] for k in range(K)] for d in runs])
    ax[1].bar(np.arange(K) + i * width, final.mean(0), width, yerr=final.std(0), label=key)
ax[1].set_xticks(np.arange(K) + 0.4 - width / 2)
ax[1].set_xticklabels(["0 (noisy)"] + [str(k) for k in range(1, K)])
ax[1].set(title="Final per-client accuracy", xlabel="Client", ylabel="Accuracy")
ax[1].legend()

# Panel 3: cosine similarity of each client's update to the averaged update (FedAvg)
base = "quick_fedavg" if use_quick else "fedavg"
key = base if base in main else list(main)[0]
for k in range(K):
    cos = np.stack([d[f"client{k}_cos_to_avg"].values for d in main[key]]).mean(0)
    ax[2].plot(main[key][0]["round"].values, cos, label=f"client {k}" + (" (noisy)" if k == 0 else ""))
ax[2].axhline(0, color="gray", lw=0.5)
ax[2].set(title=f"Update cosine to average ({key})", xlabel="Round", ylabel="Cosine")
ax[2].legend()

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/summary.png", dpi=150)
print("saved results/summary.png, groups:", list(main))

# ---------- Figure 2: noise vs no-noise control (client 0, FedAvg) ----------
ckey = next(iter(control), None)
if ckey and base in main:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("client0_cos_to_avg", "Client 0: update cosine to average"),
              ("client0_train_loss", "Client 0: training loss"),
              ("client0_acc", "Client 0: accuracy (clean local test)")]
    for i, (col, title) in enumerate(panels):
        for label, runs in [("40% noise", main[base]), ("no noise (control)", control[ckey])]:
            a = np.stack([d[col].values for d in runs])
            r = runs[0]["round"].values
            ax[i].plot(r, a.mean(0), label=f"{label} (n={len(runs)})")
            ax[i].fill_between(r, a.mean(0) - a.std(0), a.mean(0) + a.std(0), alpha=0.2)
        ax[i].set(title=title, xlabel="Round")
        ax[i].legend()
    plt.tight_layout()
    plt.savefig("results/noise_control.png", dpi=150)
    print("saved results/noise_control.png")

# ---------- Summary table: average of the last 5 rounds, mean +/- std over seeds ----------
print("\nLast-5-round average (mean +/- std over seeds)")
for key, runs in groups.items():
    def last5(col):
        return np.array([d[col].values[-5:].mean() for d in runs])
    g = last5("global_acc")
    accs = " ".join(f"{last5(f'client{k}_acc').mean():.3f}" for k in range(K))
    print(f"{key:22s} global {g.mean():.3f}+/-{g.std():.3f} | client accs {accs} | "
          f"c0 cos {last5('client0_cos_to_avg').mean():.2f} "
          f"c0 norm {last5('client0_update_norm').mean():.2f} "
          f"c0 train loss {last5('client0_train_loss').mean():.2f}")