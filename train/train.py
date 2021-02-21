import argparse
import config
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim


from collections import defaultdict
from copy import deepcopy
from model import get_loader, LSTMREgExp2, train_epoch, TSDataset
from pathlib import Path
from stratified_k_fold import get_train_val_split
from utils import seed_everything


parser = argparse.ArgumentParser(description="MagNet Parser")
parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument(
    "--n_epochs", type=int, default=30, help="number of epochs to train"
)
parser.add_argument(
    "--lr", type=float, default=0.0001, help="learning rate (default: 1e-3)"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--save_dir", default="../tmp_subm/")


parser.add_argument("--HOURS", type=int, default=72)
parser.add_argument("--n_splits", type=int, default=5)

args = parser.parse_args()

print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print("{}: {}".format(k, v))
print("========================================")


seed = args.seed
seed_everything(seed)
DATA_PATH = Path("../data/")

XCOLS = config.XCOLS
YCOLS = config.YCOLS


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH / "ours_preprocessed.csv")
    # data.timedelta = pd.to_timedelta(data.timedelta)
    # data.set_index(["period", "timedelta"], inplace=True)

    device = torch.device("cuda:1")
    n_epochs = args.n_epochs
    bs = args.bs
    lr = args.lr
    HOURS = args.HOURS
    n_splits = args.n_splits

    best_model_mse = 30
    best_model = ""

    t_begin = time.time()
    timestamp = str(t_begin).split(".")[0]

    for split, (train_idx, test_idx) in enumerate(get_train_val_split(data, n_splits)):

        if split > 0:
            break

        model = LSTMREgExp2(len(XCOLS), num_layers=1, hidden_size=512)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="sum")

        best_model_mse = 30
        best_model = ""
        train_data = data.loc[train_idx].reset_index()
        test_data = data.loc[test_idx].reset_index()

        train_ds = TSDataset(train_data[XCOLS], train_data[YCOLS], window = HOURS)
        test_ds = TSDataset(test_data[XCOLS], test_data[YCOLS], window = HOURS)

        train_dataloader = get_loader(train_ds, batch_size=bs)
        test_dataloader = get_loader(test_ds, batch_size=bs)

        for epoch in range(n_epochs):
            train_loss, val_loss = train_epoch(
                model,
                criterion,
                optimizer,
                train_dataloader,
                test_dataloader,
                device
        )

            print(
                f"Split {split } Epoch {epoch}:\n\t\
                Train loss: {train_loss:.2f}, Val loss: {val_loss:.2f}"
            )

            if val_loss < best_model_mse:
                new_file = os.path.join(
                    args.save_dir,
                    timestamp + "_" + "best-{}-{:.3f}".format(epoch, val_loss))
                best_model_mse = val_loss
                best_model = deepcopy(model)

        torch.save(
            best_model.state_dict(),
            f"{new_file}-split{split}-{val_loss:.2f}.pth",
            _use_new_zipfile_serialization=False)
