import numpy as np
import pandas as pd

import config

cols = config.SOLAR_WIND_COLS
cols_reduced = config.SOLAR_WIND_COLS_REDUCED

def prep_solar_wind_data(solar_wind):

    for col in cols:
        solar_wind[col+'_nan'] = solar_wind[col].isnull().astype(int)
    
    solar_aggs = solar_wind.groupby(
        ['period', solar_wind.index.get_level_values(1).floor("H")]
    ).agg(["mean", "std", "max", "min", "sum"])
    solar_aggs.columns = ["_".join(x) for x in solar_aggs.columns]

    cols_to_use = []

    for col in cols_reduced:
        for agg in ['mean', 'std', 'max']:
            cols_to_use.append(col+'_' + agg)

    for col in cols_reduced:
        solar_aggs[col+ '_range'] = solar_aggs[col + '_max'] - solar_aggs[col + '_min']
        cols_to_use.append(col + '_range')

    for col in cols:
        for agg in ['max', 'sum']:
            cols_to_use.append(col + '_' + 'nan' + '_' + agg)

    return solar_aggs[cols_to_use].interpolate()
