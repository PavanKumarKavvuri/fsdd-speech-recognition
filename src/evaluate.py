import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    def __init__(self, model, test_data_loader, device):
        self.model = model.to(device)
        self.test_data_loader = test_data_loader
        self.device = device

    def evaluate(self, verbose=True):
        self.model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total

        if verbose:
            print(f"\n Accuracy on test data: {accuracy:.2f}%")
            print("\n Classification Report:")
            print(classification_report(all_labels, all_preds, digits=4))
            
            print("\n Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))

        # return {
        #     "accuracy": accuracy,
        #     "predictions": np.array(all_preds),
        #     "labels": np.array(all_labels)
        # }

    def evaluate_quant_model(self, verbose=True):
        self.model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total

        if verbose:
            print(f"\n Accuracy on test data: {accuracy:.2f}%")
            print("\n Classification Report:")
            print(classification_report(all_labels, all_preds, digits=4))
            
            print("\n Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
