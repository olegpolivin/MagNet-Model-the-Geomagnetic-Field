import config
import pandas as pd
import pickle
import time
import torch


from model import get_loader, LSTMREgExp2, TSDatasetEval
from pathlib import Path
from stratified_k_fold import get_train_val_split
from utils import seed_everything

seed = 0
seed_everything(seed)
DATA_PATH = Path("../data/")

XCOLS = config.XCOLS
YCOLS = config.YCOLS


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH / "ours_preprocessed.csv")
    # data.timedelta = pd.to_timedelta(data.timedelta)
    # data.set_index(["period", "timedelta"], inplace=True)

    device = torch.device("cuda:0")
    HOURS = 72
    n_splits = 5

    with open("scaler_y.pck", "rb") as f:
        scaler_y = pickle.load(f)

    t_begin = time.time()
    timestamp = str(t_begin).split(".")[0]

    for split, (train_idx, test_idx) in enumerate(get_train_val_split(
            data, n_splits)):

        if split > 0:
            break

        model = LSTMREgExp2(len(XCOLS), num_layers=1, hidden_size=512)
        model = model.to(device)

        test_data = data.loc[test_idx].reset_index()
        test_ds = TSDatasetEval(
            test_data["period"].tolist(),
            test_data[XCOLS],
            test_data[YCOLS],
            window=HOURS,
        )

        test_dataloader = get_loader(test_ds, batch_size=128)

        model.load_state_dict(
            torch.load("../tmp_subm/1612865371_best-20-9.177-split0-9.77.pth")
        )

        period = []
        y_true0 = []
        y_true1 = []
        y_pred0 = []
        y_pred1 = []

        model.eval()
        with torch.no_grad():
            for periodcol, x, y in test_dataloader:

                period.extend(periodcol)
                x = x.to(device)

                y = y.detach().cpu().numpy()
                y = scaler_y.inverse_transform(y)
                y_true0.extend(y[:, 0])
                y_true1.extend(y[:, 1])

                predict = model.forward(x)
                predict_numpy = predict.detach().cpu().numpy()
                predict_numpy = scaler_y.inverse_transform(predict_numpy)
                y_pred0.extend(predict_numpy[:, 0])
                y_pred1.extend(predict_numpy[:, 1])

    df_pred = pd.DataFrame()
    df_pred["period"] = period
    df_pred["y_true_Y0"] = y_true0
    df_pred["y_true_Y1"] = y_true1
    df_pred["y_pred_Y0"] = y_pred0
    df_pred["y_pred_Y1"] = y_pred1
    df_pred.to_csv("dst_prediction_validation.csv", index=False)
