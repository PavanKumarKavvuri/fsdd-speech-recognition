import torch
from torch.ao.quantization.observer import MinMaxObserver

class PowerOfTwoObserver(MinMaxObserver):
    def __init__(self, *args, **kwargs):
        super(PowerOfTwoObserver, self).__init__(*args, **kwargs)

    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        # Round scale to nearest power of two
        scale = scale.clone()
        scale_log2 = torch.log2(scale)
        scale_power_of_two = torch.pow(2.0, torch.round(scale_log2))
        return scale_power_of_two, zero_point