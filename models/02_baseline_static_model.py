import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin


data = ...
y_true = data['bpd_label'].values

original_feat = ['gender',
                'gestational_age',
                'birth_weight',
                'multiplicity',
                'ANS',
                'apgar_5']


X = data[original_feat].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42, stratify=y_true)

class BaselineModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_1=64, hidden_2=32, dropout=0.5, learning_rate=0.001, epochs=10):
        self.input_dim = input_dim
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _build_model(self):
        class FCN(nn.Module):
            def __init__(self, input_dim, hidden_1, hidden_2, dropout):
                super(FCN, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_1)
                self.dropout1 = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_1, hidden_2)
                self.dropout2 = nn.Dropout(dropout)
                self.fc3 = nn.Linear(hidden_2, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout1(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.fc3(x)
                return x

        self.model = FCN(self.input_dim, self.hidden_1, self.hidden_2, self.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(y_true) / sum(y_true)], dtype=torch.float32))


    def fit(self, X, y):
        self._build_model()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i in range(len(X_tensor)):
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor[i].unsqueeze(0))
                loss = self.criterion(outputs, y_tensor[i].unsqueeze(0))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(X_tensor):.4f}')

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = (torch.sigmoid(outputs) > 0.5).float().numpy()
        return predicted

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).numpy()
        return probs
    

param_grid = {
    'hidden_1': [8, 16, 32],
    'hidden_2': [8, 16, 32],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0001],
    'epochs': [25, 50, 75]
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=BaselineModel(input_dim=6), param_grid=param_grid, scoring='f1', cv=cv, verbose=3)

grid_search.fit(X, y_true)

print("Best parameters found: ", grid_search.best_params_)
print("Best F1 score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
print(f'Area Under the Curve (AUC): {roc_auc:.2f}')

