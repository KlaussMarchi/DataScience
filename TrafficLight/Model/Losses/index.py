import torch.nn as nn

class BuiltInDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss_dict, targets=None):
        # A rede nativa retorna um dict de perdas. Apenas somamos.
        return sum(loss for loss in loss_dict.values())

class CustomDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # Onde a matemática da sua futura rede customizada vai entrar
        pass

class Losses:
    options = {
        'builtin': BuiltInDetectionLoss(),
        'custom': CustomDetectionLoss()
    }

    def __new__(cls, name):
        return cls.options.get(name, cls.options['builtin'])