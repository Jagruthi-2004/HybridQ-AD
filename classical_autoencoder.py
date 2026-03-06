import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load data
train_data   = np.load("data/train_data.npy")
test_data    = np.load("data/test_data.npy")
test_labels  = np.load("data/test_labels.npy")

# Use only normal samples for training
train_labels_raw = np.load("data/train_labels.npy")
normal_train = train_data[train_labels_raw == 0]

# Convert to tensors
X_train = torch.FloatTensor(normal_train)
X_test  = torch.FloatTensor(test_data)

train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)

# Define autoencoder
class ClassicalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 8), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ClassicalAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training
losses = []
for epoch in range(50):
    epoch_loss = 0
    for (batch,) in train_loader:
        output = model(batch)
        loss = loss_fn(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.title("Classical Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("results/classical_loss.png")
plt.show()

# Anomaly scoring
model.eval()
with torch.no_grad():
    reconstructed = model(X_test)
    scores = torch.mean((X_test - reconstructed) ** 2, dim=1).numpy()

# Threshold tuning + evaluation
threshold = np.percentile(scores, 80)  # top 20% = anomaly
predictions = (scores > threshold).astype(int)

print(f"\nAUC-ROC: {roc_auc_score(test_labels, scores):.4f}")
print(f"F1 Score: {f1_score(test_labels, predictions):.4f}")
print(classification_report(test_labels, predictions, target_names=["Normal", "Anomaly"]))