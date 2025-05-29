import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, random_split

# Optional: for classification report
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# FIXED RANDOM SEED
# ----------------------------
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# 1. Generate Synthetic Data
# ----------------------------
num_samples = 1000
seq_length = 10
input_dim = 1
threshold = 5

X = np.random.rand(num_samples, seq_length, input_dim)
y = np.sum(X[:, :, 0], axis=1) > threshold
y = y.astype(int)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split into train and validation (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# Data loaders
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ----------------------------
# 2. Define LSTM Model
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]    # take last time step
        out = self.fc(out)
        return out

# ----------------------------
# 3. Instantiate Model, Loss & Optimizer
# ----------------------------
model = LSTMClassifier(input_dim=input_dim,
                       hidden_dim=32,
                       num_layers=1,
                       output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4. Train the Model with Validation
# ----------------------------
num_epochs = 20

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    total_train_acc = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_batch).float().mean()
        total_train_loss += loss.item()
        total_train_acc += acc.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_acc / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_acc = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_batch).float().mean()

            total_val_loss += loss.item()
            total_val_acc += acc.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)

    print(f"Epoch {epoch+1}: "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Train Acc: {avg_train_acc*100:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {avg_val_acc*100:.2f}%")

# ----------------------------
# 5. Evaluate: Confusion Matrix & Classification Report
# ----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in DataLoader(dataset, batch_size=32):
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

# ----------------------------
# 5. Optional: Predict on Custom Sequences
# ----------------------------
def predict_sequence(model, sequence):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # add batch dim
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(prob).item()
        confidence = prob[0][predicted_class].item()
    return predicted_class, confidence

# Example usage:
test_seq = np.array([
    [-1.0],
    [0.2],
    [0.3],
    [0.4],
    [0.5],
    [0.6],
    [0.7],
    [0.8],
    [0.9],
    [1.0]
])
pred_class, confidence = predict_sequence(model, test_seq)
print(f"\nPredicted class: {pred_class}, Confidence: {confidence:.2f}")