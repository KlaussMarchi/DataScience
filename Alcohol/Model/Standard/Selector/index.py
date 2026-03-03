from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ModelSelector:
    options = {
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {
                'C': [0.1, 0.25, 0.5, 0.75, 1, 3, 5, 7, 10, 100, 500],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [50, 100, 200, 500, 800, 1000, 1500, 2000, 3000, 4000, 10000, 15000]
            }
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [2, 3, 5, 7, 9, 12, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'svm': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(n_estimators=300, random_state=42),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10]
            }
        },
        'naive_bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
    }

    def get(self):
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', self.selected['model'])
        ])
        
        raw_params = self.selected['params']
        params = {f'model__{k}': v for k, v in raw_params.items()}
        return model, params

    def __init__(self, name: str):
        self.chosen   = name
        self.selected = self.options[name]
