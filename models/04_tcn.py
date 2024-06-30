import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
import itertools


original_feat = ['gender', 
            'gestational_age', 
            'birth_weight', 
            'multiplicity', 
            'ANS', 
            'apgar_5']

temporal_data = pd.read_pickle(...)
data = pd.read_pickle(...)
data = data[[*original_feat, 'bpd_label']]
data = data.merge(temporal_data, left_index=True, right_index=True)

batch_size = 6

y_true = data['bpd_label'].values
X = data.drop(columns=['bpd_label']).values


X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42, stratify=y_true)

def get_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size,
                                 stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                       nn.BatchNorm1d(out_channels),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Ensemble(nn.Module):
    def __init__(self, tcn_num_channels, tcn_kernel_size, tcn_dropout, hidden_1, hidden_2, dropout):
        super(Ensemble, self).__init__()
        self.tcn = TCN(num_inputs=1, num_channels=tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)

        combined_dim = tcn_num_channels[-1]
        self.final_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        temporal_input = x[:, :53].unsqueeze(1) # ignore static features
        tcn_output = self.tcn(temporal_input).mean(dim=2)
        output = self.final_layer(tcn_output)
        return output
    



class TCN_gs(BaseEstimator, ClassifierMixin):
    def __init__(self, tcn_num_channels, tcn_kernel_size, tcn_dropout, hidden_1, hidden_2, dropout, learning_rate=0.001, epochs=100):
        self.tcn_num_channels = tcn_num_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_dropout
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.optimizer = None
        self.criterion = None

    def _build_model(self):
        self.model = Ensemble(self.tcn_num_channels, self.tcn_kernel_size, self.tcn_dropout, self.hidden_1, self.hidden_2, self.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(y_true) / sum(y_true)], dtype=torch.float32))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        self._build_model()

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            total_loss = 0
            self.model.train()
            num_samples = len(X_train)
            num_batches = int(np.ceil(num_samples / batch_size))
            self.optimizer.zero_grad()

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                inputs = X_tensor[start_idx:end_idx]
                targets = y_tensor[start_idx:end_idx]

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                total_loss += loss.item()

                if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == num_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            average_loss = total_loss / num_batches
            self.model.eval()

        print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {average_loss:.4f}')


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
    'tcn_num_channels': [[8, 16], [8, 16, 32], [8, 16, 32, 64]],
    'tcn_kernel_size': [2, 3, 5],
    'tcn_dropout': [0.2, 0.3, 0.4],
    'hidden_1': [0],
    'hidden_2': [0],
    'dropout': [0],
    'learning_rate': [0.001, 0.0001],
    'epochs': [50, 75, 100, 125],
}

'''
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_combinations = get_param_combinations(param_grid)

best_score = -np.inf
best_params = None

X_train_full = X.copy()
y_train_full = y_true.copy()

for params in param_combinations:
    fold_scores = []

    for train_index, val_index in tqdm(cv.split(X_train_full, y_train_full)):
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        model = TCN_gs(**params)
        model.fit(X_train, y_train, X_val, y_val)

        y_prob = model.predict(X_val)

        score = f1_score(y_val, y_prob)
        fold_scores.append(score)


    avg_score = np.mean(fold_scores)
    print(f"Params: {params}, Score: {avg_score}")

    if avg_score > best_score:
        best_score = avg_score
        best_params = params

print(f"Best Params: {best_params}, Best Score: {best_score}")

'''