import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .jaccard.index import JaccardIndex # Sua métrica customizada

class ModelNetwork:
    def __init__(self, network, img_size, classes=1, channels=3, lr=1e-4, dropout=0.1, num_filters=16):
        self.network = network
        self.img_size = img_size
        self.classes  = classes
        self.multiclass = (self.classes > 1)
        self.channels   = channels
        self.dropout    = dropout
        self.num_filters = num_filters
        self.lr = lr
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.get().to(self.device)
        
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer  = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

        self.iou = JaccardIndex(num_classes=self.classes, threshold=0.5)
    
    def get(self):
        if self.network == 'fasterrcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.classes)
            return model

        if self.network == 'retinanet':
            return torchvision.models.detection.retinanet_resnet50_fpn(weights_backbone='DEFAULT', num_classes=self.classes)

        if self.network == 'ssd':
            return torchvision.models.detection.ssd300_vgg16(weights_backbone='DEFAULT', num_classes=self.classes)

        return None