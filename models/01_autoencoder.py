import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import copy
import matplotlib.pyplot as plt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        self.activations = None
        self.gradients = None

        nn.init.xavier_uniform_(self.rnn1.weight_ih_l0)
        nn.init.orthogonal_(self.rnn1.weight_hh_l0)
        nn.init.constant_(self.rnn1.bias_ih_l0, 0)
        nn.init.constant_(self.rnn1.bias_hh_l0, 0)
        nn.init.xavier_uniform_(self.rnn2.weight_ih_l0)
        nn.init.orthogonal_(self.rnn2.weight_hh_l0)
        nn.init.constant_(self.rnn2.bias_ih_l0, 0)
        nn.init.constant_(self.rnn2.bias_hh_l0, 0)

    def forward(self, x):
        def forward_hook(module, input, output):
            self.activations = input[0]
            output[0].register_hook(self.save_grad)

        self.rnn1.register_forward_hook(forward_hook)
        self.rnn2.register_forward_hook(forward_hook)

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n[-1]

    def save_grad(self, grad):
        self.gradients = grad




class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        nn.init.xavier_uniform_(self.rnn1.weight_ih_l0)
        nn.init.orthogonal_(self.rnn1.weight_hh_l0)
        nn.init.constant_(self.rnn1.bias_ih_l0, 0)
        nn.init.constant_(self.rnn1.bias_hh_l0, 0)
        nn.init.xavier_uniform_(self.rnn2.weight_ih_l0)
        nn.init.orthogonal_(self.rnn2.weight_hh_l0)
        nn.init.constant_(self.rnn2.bias_ih_l0, 0)
        nn.init.constant_(self.rnn2.bias_hh_l0, 0)

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):

        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = self.output_layer(x)

        return x
    

class Autoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='mean').to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(1, n_epochs + 1):

        train_losses = []
        model = model.train()

        optimizer.zero_grad()

        seq_true = seq_true[0].to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred.squeeze(), seq_true.squeeze())
        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()

        model = model.eval()

        val_losses = []
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true[0].to(device)
                seq_pred = model(seq_true)
                val_loss = criterion(seq_pred.squeeze(), seq_true.squeeze())


                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                val_losses.append(val_loss)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)


        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')


    model.load_state_dict(best_model_wts)

    return model



data = []

with pd.HDFStore(fr"{os.getcwd()}\temporal_data.h5", 'r') as store:
    keys = store.keys()
    for key in tqdm(keys):
        pat_data = store[key]
        data_temp_spo2 = pat_data['SpO2']
        data_temp_pulse = pat_data['Pulse']
        data_temp_adem = pat_data['Ademfrequentie']
        data_temp_fio2 = pat_data['FiO2']


        for i in range(0, len(data_temp_spo2), 120):
            data.append(np.stack([data_temp_pulse[i: i + 120], data_temp_adem[i: i + 120], data_temp_spo2[i: i + 120], data_temp_fio2[i: i + 120]], axis=1))

def make_dataloader(data):
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    del data

    dataset = torch.utils.data.TensorDataset(tensor_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader


train_loader, val_loader = make_dataloader(data)
model = Autoencoder(120, 4, 32)
model = model.to(device)

model = train_model(model, train_loader, val_loader, 5)
#torch.save(model, fr"{os.getcwd()}\\lstm.pth")


with pd.HDFStore(fr"{os.getcwd()}\\temporal_data.h5", 'r') as store:
    keys = store.keys()
    encoded_data = dict()
    print(len(keys))
    for key in tqdm(keys):
        pat_data = store[key]
        data_temp_spo2 = pat_data['SpO2']
        data_temp_pulse = pat_data['Pulse']
        data_temp_adem = pat_data['Ademfrequentie']
        data_temp_fio2 = pat_data['FiO2']
        data = []
        encoded = []
        for i in range(0, len(data_temp_spo2), 120):
            data.append(np.stack((data_temp_pulse[i:i+120], data_temp_adem[i:i+120], data_temp_spo2[i:i+120], data_temp_fio2[i:i+120]), axis=1))

        tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        pred_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

        for batch in pred_loader:
            batch = batch[0].to(device)
            batch_pred = model.encoder(batch)
            batch_pred = batch_pred.detach().cpu().numpy()

            print(np.max(batch_pred.squeeze()))
            encoded.append(batch_pred.squeeze())

        encoded = np.concatenate(encoded, axis=0)
        encoded = encoded.reshape(-1)
        encoded_data[key] = encoded

encoded_data = pd.DataFrame(encoded_data)
encoded_data.to_pickle('encoded_temporal.pkl')