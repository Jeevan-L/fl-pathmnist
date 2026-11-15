"""Small CNN for 28x28 RGB images. No BatchNorm, because BN statistics behave badly in FedAvg."""
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    import torch
    m = CNN()
    print(m(torch.randn(4, 3, 28, 28)).shape)          # expect (4, 9)
    print(sum(p.numel() for p in m.parameters()), "parameters")