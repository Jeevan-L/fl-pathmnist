"""Run one federated experiment (FedAvg or FedProx) and log to W&B + CSV."""
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import wandb

from data import load_pathmnist, make_clients, to_tensor
from model import CNN
from fl import fl_round, evaluate

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["fedavg", "fedprox"], default="fedavg")
p.add_argument("--mu", type=float, default=0.0)        # FedProx strength (ignored for fedavg)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--rounds", type=int, default=30)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--epochs", type=int, default=1)         # local epochs per round
p.add_argument("--alpha", type=float, default=0.3)      # Dirichlet non-IID strength
p.add_argument("--clients", type=int, default=5)
p.add_argument("--noisy_client", type=int, default=0)
p.add_argument("--noise_rate", type=float, default=0.4)
p.add_argument("--quick", action="store_true")          # tiny CPU test
p.add_argument("--project", default="fl-pathmnist")
args = p.parse_args()

mu = args.mu if args.algo == "fedprox" else 0.0
if args.quick:
    args.rounds = 2
    args.project = "fl-pathmnist-test"                  # keep the real project clean

# Run name, e.g. fedavg_seed0 or fedprox_mu0.01_seed0
name = f"fedavg_seed{args.seed}" if mu == 0 else f"fedprox_mu{mu}_seed{args.seed}"
if args.quick:
    name = "quick_" + name

# Seeds: same seed -> same data split, noise and model init for FedAvg and FedProx
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
gen = torch.Generator().manual_seed(args.seed)

# Data
X_tr, y_tr, X_te, y_te = load_pathmnist(quick=args.quick, seed=args.seed)
clients = make_clients(X_tr, y_tr, args.clients, args.alpha, args.seed,
                       args.noisy_client, args.noise_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
for c in clients:                                  # move every client tensor to the GPU (if any)
    for key in c:
        c[key] = c[key].to(device)
x_test = to_tensor(X_te).to(device)                # official test set
y_test = torch.from_numpy(y_te).long().to(device)

model = CNN().to(device)
wandb.init(project=args.project, name=name, config=vars(args))

rows = []
for r in range(1, args.rounds + 1):
    train_losses, norms, cos = fl_round(model, clients, mu, args.lr, args.epochs, args.bs, gen)

    g_acc, g_loss = evaluate(model, x_test, y_test)
    log = {"round": r, "global_acc": g_acc, "global_loss": g_loss}
    accs = []
    for k, c in enumerate(clients):
        acc, loss = evaluate(model, c["x_test"], c["y_test"])   # global model on client's clean test
        accs.append(acc)
        log[f"client{k}_acc"] = acc
        log[f"client{k}_loss"] = loss
        log[f"client{k}_train_loss"] = train_losses[k]
        log[f"client{k}_update_norm"] = norms[k]
        log[f"client{k}_cos_to_avg"] = cos[k]
    log["clients_mean_acc"] = float(np.mean(accs))

    wandb.log(log, step=r)
    rows.append(log)
    print(f"[{name}] round {r}/{args.rounds}  global_acc={g_acc:.3f}  "
          f"client_accs={[round(a, 2) for a in accs]}  cos={[round(c, 2) for c in cos]}")

# Save CSV backup
os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv(f"results/{name}.csv", index=False)
wandb.finish()