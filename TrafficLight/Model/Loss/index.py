import torch
import torch.nn as nn
import torchvision

# 1. O seu rastreador
class LossAverager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 2. A Loss customizada (Dice)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# 3. A FÁBRICA DE LOSSES (A classe que une tudo)
class Loss:
    def __new__(cls, name, **kwargs):
        if name == 'averager':
            return LossAverager()
            
        if name == 'bce':
            return nn.BCEWithLogitsLoss(**kwargs)
            
        if name == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        
        if name == 'l1':
            return nn.SmoothL1Loss(**kwargs)
            
        if name == 'focal':
            return torchvision.ops.sigmoid_focal_loss
            
        if name == 'dice':
            return DiceLoss(**kwargs)
            
        return None