import torch.nn as nn
import torch

class StaticQuantizableModel(nn.Module):
    def __init__(self, float_model):
        super(StaticQuantizableModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = float_model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def calibrate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_tensor = batch[0]
            input_tensor = input_tensor.to('cpu')

            model(input_tensor)
