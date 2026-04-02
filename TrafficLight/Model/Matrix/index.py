import numpy as np
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    def __init__(self, num_classes, score_threshold=0.5, iou_threshold=0.5):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

    def update(self, outputs, targets):
        for out, tgt in zip(outputs, targets):
            mask = out['scores'] >= self.score_threshold
            pred_boxes  = out['boxes'][mask]
            pred_labels = out['labels'][mask]
            gt_boxes  = tgt['boxes']
            gt_labels = tgt['labels']
            
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
                
            if len(pred_boxes) == 0:
                for gl in gt_labels:
                    self.matrix[gl.item()][0] += 1
                continue
                
            if len(gt_boxes) == 0:
                for pl in pred_labels:
                    self.matrix[0][pl.item()] += 1
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes)
            
            matched_preds = set()
            for gt_idx, gl in enumerate(gt_labels):
                best_pred_idx = iou_matrix[:, gt_idx].argmax().item()
                best_iou = iou_matrix[best_pred_idx, gt_idx].item()
                
                if best_iou >= self.iou_threshold:
                    pl = pred_labels[best_pred_idx].item()
                    self.matrix[gl.item()][pl] += 1 
                    matched_preds.add(best_pred_idx)
                else:
                    self.matrix[gl.item()][0] += 1 
            
            for pred_idx, pl in enumerate(pred_labels):
                if pred_idx not in matched_preds:
                    self.matrix[0][pl.item()] += 1
    
    def plot(self):
        class_names = [str(c) for c in range(self.num_classes)]
        matrix_data = self.matrix
        matrix_norm = matrix_data.astype('float') / (matrix_data.sum(axis=1)[:, np.newaxis] + 1e-6)
        sns.heatmap(matrix_norm, annot=matrix_data, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12, "weight": "bold"})
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label (Ground Truth)')