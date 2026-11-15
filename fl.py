"""Federated learning core: local training (FedAvg / FedProx), evaluation, one communication round."""
import copy
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def evaluate(model, x, y, bs=512):
    """Return (accuracy, mean cross-entropy loss) of `model` on (x, y)."""
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            out, yb = model(x[i:i + bs]), y[i:i + bs]
            loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
            correct += (out.argmax(1) == yb).sum().item()
    return correct / len(y), loss_sum / len(y)


def local_train(model, client, mu, lr, epochs, bs, gen):
    """Train `model` on one client's data. If mu > 0 this is FedProx, if mu == 0 it is FedAvg.
    Returns the average cross-entropy training loss."""
    # Frozen copy of the global weights at the start of the round (used by the FedProx term)
    global_params = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = client["x_train"], client["y_train"]
    model.train()
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen).to(x.device)   # shuffle each epoch
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            ce = F.cross_entropy(model(x[idx]), y[idx])
            loss = ce
            if mu > 0:                                     # FedProx: penalise drifting from global
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = ce + (mu / 2) * prox
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(ce.item())                       # log the plain CE, not the prox term
    return sum(losses) / len(losses)


def fl_round(global_model, clients, mu, lr, epochs, bs, gen):
    """One communication round. Updates `global_model` in place.
    Returns per-client train loss, update norm, and cosine similarity to the averaged update."""
    g_vec = parameters_to_vector(global_model.parameters()).detach()   # global weights as 1 vector
    deltas, sizes, train_losses = [], [], []
    for c in clients:
        local = copy.deepcopy(global_model)                # each client starts from the global model
        train_losses.append(local_train(local, c, mu, lr, epochs, bs, gen))
        deltas.append(parameters_to_vector(local.parameters()).detach() - g_vec)  # update = w_k - w_g
        sizes.append(len(c["y_train"]))

    # Weighted average of updates (weights = client sample counts). Same as averaging the weights.
    w = torch.tensor(sizes, dtype=torch.float32)
    w = w / w.sum()
    avg_delta = sum(wi * d for wi, d in zip(w, deltas))
    vector_to_parameters(g_vec + avg_delta, global_model.parameters())

    norms = [d.norm().item() for d in deltas]
    cos = [F.cosine_similarity(d, avg_delta, dim=0).item() for d in deltas]
    return train_losses, norms, cos