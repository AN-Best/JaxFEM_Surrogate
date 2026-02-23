import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), 'DataSet_data', 'dataset.npz')
data = np.load(data_path)

X_theta = torch.tensor(data['X_theta'], dtype=torch.float32).unsqueeze(1)  # (N, 1, 60, 30)
X_load = torch.tensor(data['X_load'], dtype=torch.float32)                 # (N,)
X_vf = torch.tensor(data['X_vf'], dtype=torch.float32)                     # (N,)
y = torch.tensor(data['y_compliance'], dtype=torch.float32)

print(torch.isnan(X_theta).any())
print(torch.isnan(X_load).any())
print(torch.isnan(X_vf).any())
print(torch.isnan(y).any())
print(f"y min: {y.min():.4f}, max: {y.max():.4f}, mean: {y.mean():.4f}")

# Normalize targets
y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / y_std
print(f"y_std: {y_std:.4f}")

# Normalize scalar inputs
x_load_mean, x_load_std = X_load.mean(), X_load.std()
X_load_norm = (X_load - x_load_mean) / x_load_std

x_vf_mean, x_vf_std = X_vf.mean(), X_vf.std()
X_vf_norm = (X_vf - x_vf_mean) / x_vf_std


class TopoDataset(Dataset):
    def __init__(self, theta, load, vf, compliance):
        self.theta = theta
        self.load = load
        self.vf = vf
        self.compliance = compliance

    def __len__(self):
        return len(self.compliance)

    def __getitem__(self, idx):
        return self.theta[idx], self.load[idx], self.vf[idx], self.compliance[idx]


dataset = TopoDataset(X_theta, X_load_norm, X_vf_norm, y_norm)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)


class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (16, 30, 15)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 15, 7)
            nn.Flatten()                                  # (32*15*7,)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*15*7 + 2, 128),                 # +2 for x_load and vf
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, theta, x_load, vf):
        features = self.cnn(theta)
        x = torch.cat([features, x_load.unsqueeze(1), vf.unsqueeze(1)], dim=1)
        return self.fc(x).squeeze(1)


model = SurrogateModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    model.train()
    train_loss = 0.
    for theta, x_load, vf, compliance in train_loader:
        optimizer.zero_grad()
        pred = model(theta, x_load, vf)
        loss = loss_fn(pred, compliance)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for theta, x_load, vf, compliance in test_loader:
            pred = model(theta, x_load, vf)
            test_loss += loss_fn(pred, compliance).item()

    print(f"Epoch {epoch+1:3d} | train loss: {train_loss/len(train_loader):.4f} | test loss: {test_loss/len(test_loader):.4f}")

# Save model, normalization stats for all inputs
torch.save({
    'model': model.state_dict(),
    'y_mean': y_mean,
    'y_std': y_std,
    'x_load_mean': x_load_mean,
    'x_load_std': x_load_std,
    'x_vf_mean': x_vf_mean,
    'x_vf_std': x_vf_std,
}, os.path.join(os.path.dirname(__file__), 'surrogate.pt'))

print("Model saved.")