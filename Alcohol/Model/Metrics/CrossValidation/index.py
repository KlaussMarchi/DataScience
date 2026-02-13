import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import clone
import pandas as pd


class CrossValidation:
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted', 
        'recall':  'recall_weighted',
        'roc_auc': 'roc_auc_ovr_weighted', 
    }
    
    def __init__(self, model, xData, yData, k=4, seed=42):
        self.model = clone(model)
        self.xData = xData
        self.yData = yData
        self.scores = {}
        self.seed   = seed
        self.k = k

    def update(self):
        self.cv = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        result = cross_validate(**{
            'estimator': self.model,
            'X':  self.xData, 'y':  self.yData,
            'cv': self.cv,
            'scoring': self.scoring,
            'return_train_score': False
        })

        self.scores   = [self.process(result, metric) for metric in self.scoring.keys()]
        self.accuracy  = float(np.mean(self.scores[0]['values']))
        self.precision = float(np.mean(self.scores[1]['values']))
        self.recall = float(np.mean(self.scores[2]['values']))
        self.auc    = float(np.mean(self.scores[3]['values']))

        data = []
        for metric in self.scores:
            row = metric.copy()
            row.update({f'k{i}': value for i, value in enumerate(metric.get('values'))})
            data.append(row)
        
        self.df = pd.DataFrame(data)

    def process(self, result, key):
        values = result[f'test_{key}']

        return {
            'name': key,
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def info(self):
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'auc': self.auc,
        }

    def print(self):        
        for metric in self.scores:
            name = metric.get('name').upper()
            mean = metric.get('mean')*100
            std  = metric.get('std')
            print(f'{name}: {mean:.2f}% (±{std*100:.2f}%)')

        return self.df

    def plot(self):
        plt.figure(figsize=(20, 4))

        for i, row in self.df.iterrows():
            metric, values = row['name'], row['values']
            kData = [j+1 for j in range(self.k)]

            plt.subplot(1, len(self.df), i+1)
            plt.plot(kData, values)
            plt.scatter(kData, values, s=20)
            for info in ['mean', 'std']:
                plt.scatter([], [], label=f'{info}: {row[info]:.3f}')
            plt.xticks(kData); plt.title(f'{metric.title()}')
            plt.grid(); plt.xlabel('k'); plt.ylabel(metric); plt.legend()

        plt.show()