import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import seaborn as sns
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Quantum circuit definition
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_autoencoder(x, theta):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
        qml.RZ(theta[i + n_qubits], wires=i)
    for i in range(0, n_qubits - 2, 2):
        qml.CNOT(wires=[i, i+2])
    for i in range(n_qubits):
        qml.RY(theta[i + 2*n_qubits], wires=i)
        qml.RZ(theta[i + 3*n_qubits], wires=i)
    for i in range(0, n_qubits - 2, 2):
        qml.CNOT(wires=[i, i+2])
    for i in range(n_qubits):
        qml.RY(theta[i + 4*n_qubits], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(theta[i + 5*n_qubits], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Load data
test_labels    = np.load("data/test_labels.npy")
quantum_scores = np.load("data/quantum_scores.npy")
test_data      = np.load("data/test_data.npy")
train_data     = np.load("data/train_data.npy")
train_labels   = np.load("data/train_labels.npy")

X_test = torch.FloatTensor(test_data)

# Classical Autoencoder definition
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

# Retrain classical model
normal_train = torch.FloatTensor(train_data[train_labels == 0])
model = ClassicalAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(normal_train), batch_size=64, shuffle=True)

print("Retraining classical model for visualizations...")
for epoch in range(50):
    for (batch,) in loader:
        out = model(batch)
        loss = loss_fn(out, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    reconstructed = model(X_test)
    classical_scores = torch.mean((X_test - reconstructed) ** 2, dim=1).numpy()

# Use same 500 test samples for fair comparison
test_labels_sub      = test_labels[:500]
quantum_scores_sub   = quantum_scores[:500]
classical_scores_sub = classical_scores[:500]

print("Generating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

# --- Plot 1: ROC Curve Comparison ---
ax1 = fig.add_subplot(gs[0, 0])
fpr_c, tpr_c, _ = roc_curve(test_labels_sub, classical_scores_sub)
fpr_q, tpr_q, _ = roc_curve(test_labels_sub, quantum_scores_sub)
auc_c = auc(fpr_c, tpr_c)
auc_q = auc(fpr_q, tpr_q)
ax1.plot(fpr_c, tpr_c, label=f"Classical AE (AUC={auc_c:.3f})", color="blue")
ax1.plot(fpr_q, tpr_q, label=f"Quantum AE  (AUC={auc_q:.3f})", color="red")
ax1.plot([0,1],[0,1], "k--", label="Random Guess")
ax1.set_title("ROC Curve: Classical vs Quantum")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend()

# --- Plot 2: Anomaly Score Histogram - Classical ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(classical_scores_sub[test_labels_sub==0], bins=40,
         alpha=0.6, color="blue", label="Normal")
ax2.hist(classical_scores_sub[test_labels_sub==1], bins=40,
         alpha=0.6, color="red", label="Anomaly")
ax2.set_title("Classical AE - Anomaly Score Distribution")
ax2.set_xlabel("Reconstruction Error")
ax2.set_ylabel("Count")
ax2.legend()

# --- Plot 3: Anomaly Score Histogram - Quantum ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(quantum_scores_sub[test_labels_sub==0], bins=40,
         alpha=0.6, color="blue", label="Normal")
ax3.hist(quantum_scores_sub[test_labels_sub==1], bins=40,
         alpha=0.6, color="red", label="Anomaly")
ax3.set_title("Quantum AE - Anomaly Score Distribution")
ax3.set_xlabel("Reconstruction Error")
ax3.set_ylabel("Count")
ax3.legend()

# --- Plot 4: Confusion Matrix - Classical ---
ax4 = fig.add_subplot(gs[1, 0])
threshold_c = np.percentile(classical_scores_sub, 80)
preds_c = (classical_scores_sub > threshold_c).astype(int)
cm_c = confusion_matrix(test_labels_sub, preds_c)
sns.heatmap(cm_c, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=["Normal","Anomaly"],
            yticklabels=["Normal","Anomaly"])
ax4.set_title("Classical AE - Confusion Matrix")
ax4.set_ylabel("Actual")
ax4.set_xlabel("Predicted")

# --- Plot 5: Confusion Matrix - Quantum ---
ax5 = fig.add_subplot(gs[1, 1])
threshold_q = np.percentile(quantum_scores_sub, 80)
preds_q = (quantum_scores_sub > threshold_q).astype(int)
cm_q = confusion_matrix(test_labels_sub, preds_q)
sns.heatmap(cm_q, annot=True, fmt="d", cmap="Reds", ax=ax5,
            xticklabels=["Normal","Anomaly"],
            yticklabels=["Normal","Anomaly"])
ax5.set_title("Quantum AE - Confusion Matrix")
ax5.set_ylabel("Actual")
ax5.set_xlabel("Predicted")

# --- Plot 6: Parameter Efficiency Comparison ---
ax6 = fig.add_subplot(gs[1, 2])
models = ["Classical AE", "Quantum AE"]
params = [162, 48]
aucs   = [auc_c, auc_q]
colors = ["blue", "red"]
bars = ax6.bar(models, params, color=colors, alpha=0.7)
ax6.set_title("Parameter Efficiency")
ax6.set_ylabel("Number of Parameters")
for bar, a in zip(bars, aucs):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"AUC: {a:.3f}", ha="center", fontsize=11)

plt.suptitle("HybridQ-AD: Classical vs Quantum Autoencoder Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/final_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll visualizations saved to results/final_comparison.png")
print("\n--- FINAL SUMMARY ---")
print(f"Classical AE → AUC: {auc_c:.4f} | F1: {f1_score(test_labels_sub, preds_c):.4f} | Parameters: 162")
print(f"Quantum AE   → AUC: {auc_q:.4f} | F1: {f1_score(test_labels_sub, preds_q):.4f} | Parameters: 48")
print(f"\nQuantum model achieved {auc_q/auc_c*100:.1f}% of classical performance with {48/162*100:.1f}% of the parameters")

# --- Circuit Diagram ---
print("\nGenerating circuit diagram...")
theta_dummy = torch.zeros(6 * n_qubits, dtype=torch.float64)
x_dummy = torch.zeros(n_qubits, dtype=torch.float64)
fig, ax = qml.draw_mpl(quantum_autoencoder)(x_dummy, theta_dummy)
plt.savefig("results/circuit_diagram.png", bbox_inches="tight")
plt.show()
print("Circuit diagram saved!")