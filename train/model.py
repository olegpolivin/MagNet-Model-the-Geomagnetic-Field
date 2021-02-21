import numpy as np
import pickle
import torch
import torch.nn as nn

from torch.utils.data import Dataset

with open("scaler_y.pck", "rb") as f:
    scaler_y = pickle.load(f)

class TSDataset(Dataset):
 
    def __init__(self, inputs, targets, window):

        self.data =  torch.tensor(inputs.values.astype(np.float32))
        self.window = window
        self.targets = targets.values.astype(np.float32)

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.targets[index+self.window]
        return x, y
 
    def __len__(self):
        return len(self.data) - self.window

class TSDatasetEval(Dataset):
 
    def __init__(self, periods, inputs, targets, window):

        self.periods = periods
        self.data =  torch.tensor(inputs.values.astype(np.float32))
        self.window = window
        self.targets = targets.values.astype(np.float32)

    def __getitem__(self, index):
        period = self.periods[index+self.window]
        x = self.data[index:index+self.window]
        y = self.targets[index+self.window]
        return period, x, y
 
    def __len__(self):
        return len(self.data) - self.window

def get_loader(df, batch_size):
    loader = torch.utils.data.DataLoader(
        df,
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True)

    return loader

class LSTMREgExp2(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers = 1,
        hidden_size = 512):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout=0.4)

        self.conv1d = nn.Sequential(
            nn.MaxPool1d(3),
            nn.Dropout(0.25),
            nn.ReLU())

        self.logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72*170, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 2))

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        # print(lstm_out.shape)
        conv1d = self.conv1d(lstm_out)
        # print(conv1d.shape)
        t = self.logits(conv1d)
        return t

def train_epoch(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device):

    epoch_loss = 0
    num_samples = 0
    model.train()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        predict = model.forward(x)
        loss = criterion(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_samples += x.shape[0]
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / (2 * num_samples)

    val_loss = 0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            y = y.detach().cpu().numpy()
            y = torch.tensor(scaler_y.inverse_transform(y))
            #y = y.to(device)

            predict = model.forward(x)
            predict_numpy = predict.detach().cpu().numpy()

            predict_numpy = scaler_y.inverse_transform(predict_numpy)
            predict_numpy = torch.tensor(predict_numpy)#.to(device)

            loss = criterion(predict_numpy, y)
            num_samples += x.shape[0]
            val_loss += loss.item()

    val_loss = val_loss / (2 * num_samples)
    return epoch_loss**.5, val_loss**.5
