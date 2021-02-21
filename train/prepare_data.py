import numpy as np
import pandas as pd
import pickle


from pathlib import Path
from process_solar_wind import prep_solar_wind_data
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("../data/")

dst = pd.read_csv(DATA_PATH / "dst_labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)

solar_wind = pd.read_csv(DATA_PATH / "solar_wind.csv")
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

sunspots = pd.read_csv(DATA_PATH / "sunspots.csv")
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)

sw = prep_solar_wind_data(solar_wind)
swsun = pd.merge(sw, sunspots, how='left', left_index=True, right_index=True)
swsun = swsun.interpolate()

scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.fit(swsun)
scaler_y.fit(dst)

features_scaled = pd.DataFrame(
    scaler_x.transform(swsun),
    index = swsun.index,
    columns = swsun.columns
    
)

y_scaled = pd.DataFrame(
    scaler_y.transform(dst),
    index = dst.index,
    columns = dst.columns
    
)

features = pd.merge(y_scaled, features_scaled, left_index=True, right_index=True)
features['YO'] = features['dst']
features['Y1'] = features['dst'].shift(-1)
features = features.dropna()

with open("scaler_x.pck", "wb") as f:
    pickle.dump(scaler_x, f)

with open("scaler_y.pck", "wb") as f:
    pickle.dump(scaler_y, f)

features.to_csv(DATA_PATH / 'ours_preprocessed.csv')