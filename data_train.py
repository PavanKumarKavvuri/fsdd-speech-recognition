import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torchaudio.transforms as T
import numpy as np
import torch.nn.functional as F
import torch
import torchaudio
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import random
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



# 1. Prepare data list with (audio_path, label)
def prepare_data_list(root_dir):
    data = []
    for filename in os.listdir(root_dir)[:]:
        if filename.endswith('.wav'):
            label = int(filename.split('_')[0])  # Extract label from filename
            path = os.path.join(root_dir, filename)
            data.append((path, label))
    return data

# 2. Dataset class
class SpokenDigitDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, denoise_data=False, sample_rate=8000):
        self.data = data
        self.transform = transform
        self.denoise_data = denoise_data
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,         
            n_mfcc=13,                 
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40
            }
        )

    def __len__(self):
        return len(self.data)

    def apply_bandpass_filter(self, waveform, sample_rate):
        """
        Applies bandpass filter (300–3000 Hz) and gain normalization to the waveform.
        Returns the processed waveform.
        """
        effects = [
            ['bandpass', '300', '3000'],  # speech frequency range
            ['gain', '-n']                # normalize to 0 dB
        ]
        processed_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )

        return processed_waveform

    def pad_or_trim_mfcc(self, mfcc, max_len=36):
        """
        Pads or trims the MFCC array along the time axis to make its shape (n_mfcc, max_len).
        
        Parameters:
            mfcc (np.ndarray): MFCC feature array of shape (n_mfcc, time_steps)
            max_len (int): Desired number of time steps

        Returns:
            np.ndarray: MFCC array of shape (n_mfcc, max_len)
        """
        _, _, time_steps = mfcc.shape

        pad_width = max_len - time_steps

        if time_steps < max_len:
            # # Pad with zeros at the end
            padded_mfcc = F.pad(mfcc, (0, pad_width))
            return padded_mfcc
        
        elif time_steps > max_len:
            # # Trim to max_len
            trimmed_mfcc = mfcc[:, :, :max_len]
            return trimmed_mfcc
        
        return mfcc  # Already the correct size

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]

        waveform, samplerate = torchaudio.load(audio_path)
        if self.denoise_data:
            waveform = self.apply_bandpass_filter(waveform, samplerate)
        waveform = waveform / waveform.abs().max()

        mfcc_features = self.mfcc_transform(waveform)
        updated_mfcc_features = self.pad_or_trim_mfcc(mfcc_features)

        return updated_mfcc_features, label

# ----------------------------
# 2. Define LSTM Model
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0
                            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, 1, 13, 36) → we need (batch_size, 36, 13)
        x = x.squeeze(1).permute(0, 2, 1)  # (B, 13, 36) → (B, 36, 13)

        # LSTM output
        out, (hn, cn) = self.lstm(x)  # out: (B, 36, H), hn: (num_layers, B, H)

        # Use the last hidden state for classification
        out = self.fc(hn[-1])  # hn[-1] = (B, hidden_size)

        return out



# 3. Use it:
root_dir = '/home/pavan/Music/spectrum/free-spoken-digit-dataset/recordings'
data_list = prepare_data_list(root_dir)
# print(data_list)

train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
# print(train_data)

train_dataset = SpokenDigitDataset(train_data)
test_dataset = SpokenDigitDataset(test_data)

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

# 4. Use DataLoader for batching:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


input_dim = 13  # Number of MFCC coefficients
hidden_dim = 38 # 20 # 32 # 128  # Hidden dimension for LSTM
num_layers = 1  # Number of LSTM layers
output_dim = 10  # Number of classes (digits 0-9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       output_dim=output_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate training accuracy at the end of the epoch
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Training Accuracy: {accuracy:.2f}%")




correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # print(f'Predicted: {predicted}, Labels: {labels}')
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(outputs.shape)      # Should be [batch_size, num_classes]
    print(labels.shape)       # Should be [batch_size]
    print(predicted[:5])      # Check predicted class indices
    print(labels[:5])         # Check actual class indices


accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')

print(classification_report(all_labels, all_preds, digits=4))

layer_params = defaultdict(str)
for name, param in model.named_parameters():
    layer_num = str(name)## int(name.split("_")[-1][1:])  # Extract layer number
    layer_params[layer_num] = layer_params.get(layer_num, 0) + param.numel()

print(layer_params)  # {0: 49408, 1: 33024} [1][4]

# Calculate memory
total_memory = sum(param.numel() for param in model.parameters()) * 4
total_memory_in_kb = total_memory * 1024
print(f"Total memory: {total_memory_in_kb/1e6:.2f} KB")  # 0.33 MB [2][4]

conf_matrix = confusion_matrix(all_labels, all_preds)

labels = [str(i) for i in range(10)]  # For digit classification (0–9)

fig = px.imshow(
    conf_matrix,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(x="Predicted", y="True", color="Count"),
    x=labels,
    y=labels,
    title="Confusion Matrix"
)

fig.update_layout(
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
)

# fig.show()

