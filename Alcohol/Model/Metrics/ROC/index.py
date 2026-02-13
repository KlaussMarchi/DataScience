import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.base import clone

class RocCurve:
    def __init__(self, model, xData, yData, cv):
        self.model = model
        self.cv    = cv
        
        self.x = xData.values if hasattr(xData, 'values') else np.array(xData)
        self.y = yData.values if hasattr(yData, 'values') else np.array(yData)
        
        self.classes = np.unique(self.y)
        self.is_multiclass = len(self.classes) > 2
        
        self.mean_fpr = np.linspace(0, 1, 100)
        self.tprs = []
        self.aucs = []
        self.update()

    def update(self):
        self.tprs = []
        self.aucs = []
        
        for train_idx, test_idx in self.cv.split(self.x, self.y):
            xt, xv = self.x[train_idx], self.x[test_idx]
            yt, yv = self.y[train_idx], self.y[test_idx]
            
            mdl = clone(self.model).fit(xt, yt)
            probas = mdl.predict_proba(xv)
            
            if self.is_multiclass:
                fold_auc = roc_auc_score(yv, probas, multi_class='ovr', average='weighted')
                y_bin    = label_binarize(yv, classes=self.classes)
                fpr, tpr, _ = roc_curve(y_bin.ravel(), probas.ravel())
            else:
                fold_auc    = roc_auc_score(yv, probas[:, 1])
                fpr, tpr, _ = roc_curve(yv, probas[:, 1])
            
            self.aucs.append(fold_auc)
            interp_tpr    = np.interp(self.mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            self.tprs.append(interp_tpr)

        self.auc = float(np.mean(self.aucs))
        self.std = float(np.std(self.aucs))

    def plot(self):
        mean_tpr     = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0 # Garante fim no 1
        
        plt.plot(self.mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'Mean ROC (AUC = {self.auc:.2f} ± {self.std:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
        plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC (Cross-Validation Média)')
        plt.legend(loc="lower right"); plt.grid(alpha=0.3)