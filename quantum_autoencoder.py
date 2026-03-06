import numpy as np
import pennylane as qml
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load preprocessed data
train_data  = np.load("data/train_data.npy")
test_data   = np.load("data/test_data.npy")
train_labels = np.load("data/train_labels.npy")
test_labels  = np.load("data/test_labels.npy")

# Use only normal samples for training
normal_train = train_data[train_labels == 0]

# Use a small subset to keep training fast
normal_train = normal_train[:500]

print(f"Training on {len(normal_train)} normal samples")

# Quantum device - 8 qubits
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_autoencoder(x, theta):
    # ENCODER - Angle encoding (data goes in as rotation angles)
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    # Layer 1 - Entanglement + trainable rotations
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
        qml.RZ(theta[i + n_qubits], wires=i)

    # Layer 2 - Long range entanglement
    for i in range(0, n_qubits - 2, 2):
        qml.CNOT(wires=[i, i+2])
    for i in range(n_qubits):
        qml.RY(theta[i + 2*n_qubits], wires=i)
        qml.RZ(theta[i + 3*n_qubits], wires=i)

    # DECODER - Mirror of encoder
    for i in range(0, n_qubits - 2, 2):
        qml.CNOT(wires=[i, i+2])
    for i in range(n_qubits):
        qml.RY(theta[i + 4*n_qubits], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits):
        qml.RY(theta[i + 5*n_qubits], wires=i)

    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Initialize random parameters
n_params = 6 * n_qubits
theta = torch.tensor(np.random.uniform(0, 2*np.pi, n_params),
                     requires_grad=True, dtype=torch.float64)

optimizer = optim.Adam([theta], lr=0.01)

# Training loop
print("\nStarting quantum training...")
losses = []

for epoch in range(100):
    epoch_loss = 0
    # Use small batches
    indices = np.random.choice(len(normal_train), size=10, replace=False)

    for idx in indices:
        x = torch.tensor(normal_train[idx], dtype=torch.float64)

        optimizer.zero_grad()

        # Forward pass
        output = torch.stack(quantum_autoencoder(x, theta))

        # Normalize output to same range as input
        output_norm = (output + 1) / 2 * np.pi

        # Reconstruction loss
        loss = torch.mean((x - output_norm) ** 2)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / 10
    losses.append(avg_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure()
plt.plot(losses)
plt.title("Quantum Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/quantum_loss.png")
plt.show()

# Anomaly scoring on test set
print("\nScoring test samples...")
test_subset = test_data[:500]
test_labels_subset = test_labels[:500]

scores = []
for i, x in enumerate(test_subset):
    x_tensor = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        output = torch.stack(quantum_autoencoder(x_tensor, theta))
        output_norm = (output + 1) / 2 * np.pi
        score = torch.mean((x_tensor - output_norm) ** 2).item()
    scores.append(score)
    if (i+1) % 100 == 0:
        print(f"Scored {i+1}/500 samples")

scores = np.array(scores)

# Threshold and evaluation
threshold = np.percentile(scores, 80)
predictions = (scores > threshold).astype(int)

print("\n--- QUANTUM AUTOENCODER RESULTS ---")
print(f"AUC-ROC: {roc_auc_score(test_labels_subset, scores):.4f}")
print(f"F1 Score: {f1_score(test_labels_subset, predictions):.4f}")
print(classification_report(test_labels_subset, predictions,
      target_names=["Normal", "Anomaly"]))

print("\n--- COMPARISON ---")
print("Classical AE → AUC-ROC: 0.7013 | F1: 0.5172")
print(f"Quantum AE  → AUC-ROC: {roc_auc_score(test_labels_subset, scores):.4f} | F1: {f1_score(test_labels_subset, predictions):.4f}")

# Save scores for later analysis
np.save("data/quantum_scores.npy", scores)
print("\nDone! Quantum scores saved.")