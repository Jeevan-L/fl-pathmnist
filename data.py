"""Data utilities: load PathMNIST, split it across clients (Dirichlet), add label noise."""
import numpy as np
import torch
from medmnist import PathMNIST

N_CLASSES = 9


def load_pathmnist(quick=False, seed=0):
    """Return train/test images and labels as numpy arrays."""
    train = PathMNIST(split="train", download=True, size=28)
    test = PathMNIST(split="test", download=True, size=28)
    X_tr, y_tr = train.imgs, train.labels.squeeze()   # imgs: (N, 28, 28, 3) uint8
    X_te, y_te = test.imgs, test.labels.squeeze()
    if quick:  # tiny subset so CPU tests are fast
        rng = np.random.default_rng(seed)
        i = rng.choice(len(X_tr), 3000, replace=False)
        j = rng.choice(len(X_te), 500, replace=False)
        X_tr, y_tr, X_te, y_te = X_tr[i], y_tr[i], X_te[j], y_te[j]
    return X_tr, y_tr, X_te, y_te


def to_tensor(X):
    """uint8 (N,28,28,3) -> float tensor (N,3,28,28), scaled to [-1, 1]."""
    X = torch.from_numpy(X).float() / 255.0
    X = (X - 0.5) / 0.5
    return X.permute(0, 3, 1, 2).contiguous()


def dirichlet_split(labels, n_clients, alpha, seed, min_size=50):
    """For each class, share its samples among clients using Dirichlet(alpha) proportions.
    Small alpha = very uneven (non-IID). Retry until every client has >= min_size samples."""
    rng = np.random.default_rng(seed)
    while True:
        client_idx = [[] for _ in range(n_clients)]
        for c in range(N_CLASSES):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            p = rng.dirichlet([alpha] * n_clients)           # class c's share per client
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                client_idx[k] += part.tolist()
        if min(len(i) for i in client_idx) >= min_size:
            return [np.array(i) for i in client_idx]


def add_label_noise(y, rate, seed):
    """Symmetric noise: flip `rate` of the labels to a DIFFERENT random class."""
    rng = np.random.default_rng(seed)
    y = y.copy()
    flip = rng.choice(len(y), int(rate * len(y)), replace=False)
    y[flip] = (y[flip] + rng.integers(1, N_CLASSES, size=len(flip))) % N_CLASSES
    return y


def make_clients(X, y, n_clients=5, alpha=0.3, seed=0,
                 noisy_client=0, noise_rate=0.4, test_frac=0.2):
    """Split data into clients. Each client: 80% train, 20% clean local test.
    Noise is added to the TRAIN labels of `noisy_client` only, after the split."""
    X_t = to_tensor(X)
    splits = dirichlet_split(y, n_clients, alpha, seed)
    rng = np.random.default_rng(seed + 1)
    clients = []
    for k, idx in enumerate(splits):
        idx = rng.permutation(idx)
        n_test = int(test_frac * len(idx))
        te, tr = idx[:n_test], idx[n_test:]
        y_tr_clean = y[tr]
        y_tr = add_label_noise(y_tr_clean, noise_rate, seed + 1000) if k == noisy_client else y_tr_clean
        clients.append(dict(
            x_train=X_t[tr], y_train=torch.from_numpy(y_tr).long(),
            y_train_clean=torch.from_numpy(y_tr_clean).long(),   # only for checking noise
            x_test=X_t[te], y_test=torch.from_numpy(y[te]).long(),  # always clean
        ))
    return clients


if __name__ == "__main__":
    # Quick sanity check: python data.py
    X_tr, y_tr, X_te, y_te = load_pathmnist(quick=True)
    clients = make_clients(X_tr, y_tr)
    for k, c in enumerate(clients):
        counts = np.bincount(c["y_train_clean"].numpy(), minlength=N_CLASSES)
        noisy = (c["y_train"] != c["y_train_clean"]).float().mean().item()
        print(f"client {k}: train={len(c['y_train'])} test={len(c['y_test'])} "
              f"noisy_frac={noisy:.2f} class_counts={counts}")