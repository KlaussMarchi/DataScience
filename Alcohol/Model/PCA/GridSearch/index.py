from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np


class GridSearch:
    def __init__(self, model, params, xData, yData, k=5, test_size=0.20):
        self.model  = clone(model)
        self.params = params
        self.xData  = xData
        self.yData  = yData
        self.test_size = test_size
        self.k   = k
        self.grid = None

    def update(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.xData, self.yData, 
            test_size=self.test_size, 
            stratify=self.yData, 
            random_state=42
        )
        
        is_multiclass  = len(np.unique(self.yData)) > 2
        scoring_metric = 'roc_auc_ovr_weighted' if is_multiclass else 'roc_auc'
        cv_splitter = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        self.grid = GridSearchCV(
            estimator=self.model, 
            param_grid=self.params, 
            cv=cv_splitter, 
            scoring=scoring_metric,
            n_jobs=-1, 
            verbose=1,
            return_train_score=False
        )
        
        self.grid.fit(self.xTrain, self.yTrain)

    def evaluate(self):
        best_model = self.grid.best_estimator_
        y_proba = best_model.predict_proba(self.xTest)
        
        is_multiclass = len(np.unique(self.yTest)) > 2
        if is_multiclass:
            auc = roc_auc_score(self.yTest, y_proba, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(self.yTest, y_proba[:, 1])
        
        return best_model, auc
    
