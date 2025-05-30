import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self, model, num_epochs, data_loader, device, learning_rate):
        self.model = model.to(device)
        self.epochs = num_epochs
        self.data_loader = data_loader
        self.device = device
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Evaluate training accuracy at the end of the epoch
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")
