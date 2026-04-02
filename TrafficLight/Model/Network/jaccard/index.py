from torchvision.ops import box_iou


class JaccardIndex:
    def __init__(self, num_classes, threshold=0.5):
        self.num_classes = num_classes
        self.threshold   = threshold
        self.reset()

    def reset(self):
        self.history = {c: [] for c in range(1, self.num_classes)}

    def update(self, outputs, targets):
        iouData = {c: [] for c in range(1, self.num_classes)}
        
        for out, tgt in zip(outputs, targets):
            mask = out['scores'] >= self.threshold
            pred_boxes  = out['boxes'][mask]
            pred_labels = out['labels'][mask]
            
            gt_boxes  = tgt['boxes']
            gt_labels = tgt['labels']
            
            for c in range(1, self.num_classes):
                c_pred_boxes = pred_boxes[pred_labels == c]
                c_gt_boxes   = gt_boxes[gt_labels == c]
                
                if len(c_pred_boxes) == 0 and len(c_gt_boxes) == 0:
                    continue
                    
                if len(c_pred_boxes) == 0 or len(c_gt_boxes) == 0:
                    iouData[c].append(0.0)
                    self.history[c].append(0.0)
                    continue
                    
                iou_matrix  = box_iou(c_pred_boxes, c_gt_boxes)
                max_ious, _ = iou_matrix.max(dim=0)
                val = max_ious.mean().item()
                
                iouData[c].append(val)
                self.history[c].append(val)
                
        batch_means = [sum(v)/len(v) for v in iouData.values() if len(v) > 0]
        batch_miou  = sum(batch_means) / len(batch_means) if len(batch_means) > 0 else 0.0
        return batch_miou

    def compute(self):
        class_ious = {}
        for c, vals in self.history.items():
            class_ious[c] = sum(vals) / len(vals) if len(vals) > 0 else 0.0
        
        valid_means = list(class_ious.values())
        mean_iou    = sum(valid_means) / len(valid_means) if len(valid_means) > 0 else 0.0
        return (mean_iou, class_ious)