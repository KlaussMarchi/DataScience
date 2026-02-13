import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone


class ConfusionMatrix:
    def __init__(self, model, xData, yData, cv, normalize='true'):
        self.model = clone(model)
        self.cv = cv
        self.normalize = normalize
        self.xData = xData.values if hasattr(xData, 'values') else np.array(xData)
        self.yData = yData.values if hasattr(yData, 'values') else np.array(yData)
        self.classes = np.unique(self.yData)
        
        self.y_true_all = []
        self.y_pred_all = []
        self.updated = False
        self.update()

    def update(self):
        self.y_true_all = []
        self.y_pred_all = []

        for train_idx, test_idx in self.cv.split(self.xData, self.yData):
            xt, xv = self.xData[train_idx], self.xData[test_idx]
            yt, yv = self.yData[train_idx], self.yData[test_idx]
            
            clf = clone(self.model)
            clf.fit(xt, yt)
            preds = clf.predict(xv)
        
            self.y_true_all.extend(yv)
            self.y_pred_all.extend(preds)
        
        self.updated = True

    def get(self):
        if not self.updated:
            self.update()

        counts = confusion_matrix(self.y_true_all, self.y_pred_all, labels=self.classes, normalize=None)

        if len(self.classes) == 2:
            tn, fp, fn, tp = counts.ravel()
            return {
                'TP': int(tp), 'TN': int(tn),
                'FP': int(fp), 'FN': int(fn),
                'global_accuracy': (tp + tn) / (tp + tn + fp + fn)
            }
        
        metrics = {}
        total_per_class = counts.sum(axis=1)
        
        for i, cls_name in enumerate(self.classes):
            tp    = counts[i, i]
            total = total_per_class[i]
            acc = tp / total if total > 0 else 0.0
            metrics[f'Acc_{cls_name}'] = f"{acc*100:.1f}% ({tp}/{total})"
        
        return metrics

    def plot(self):
        if not self.updated:
            self.update()
        
        cm_display = confusion_matrix(self.y_true_all, self.y_pred_all, labels=self.classes, normalize=self.normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=self.classes)
        fmt  = '.1%' if self.normalize else 'd'
        
        disp.plot(cmap='Blues', ax=plt.gca(), values_format=fmt, colorbar=False)
        plt.title(f'Confusion Matrix')
        plt.grid(False)