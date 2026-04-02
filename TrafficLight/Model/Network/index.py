import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
import numpy as np

from .jaccard.index import JaccardIndex


class Network:
    def __init__(self, name='', img_size=256, classes=1, channels=3, dropout=0.1, num_filters=32, lr=0.001):
        self.name   = name
        self.img_size = img_size
        self.classes  = classes
        self.multiclass  = (self.classes > 1)
        self.channels    = channels
        self.dropout     = dropout
        self.num_filters = num_filters
        self.lr  = lr
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.get_model()
        self.model  = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.iou = JaccardIndex(num_classes=self.classes, threshold=0.5)
        
    def get_model(self):
        if self.name == 'fasterrcnn':
            model       = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.classes)
            return model

        if self.name == 'retinanet':
            return torchvision.models.detection.retinanet_resnet50_fpn(weights='DEFAULT')

        if self.name == 'ssd':
            return torchvision.models.detection.ssd300_vgg16(weights='DEFAULT')

        if self.name == 'maskrcnn':
            return torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

        return None