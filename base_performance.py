import pandas as pd
import numpy as np
from datasetsforecast.m4 import M4, M4Info
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_scaled_error


if __name__ == "__main__":
    group = M4Info['Monthly']
    Y_df, _, S_df = M4.load(directory='data', group=group.name)
    series_limit = 5000

    unique_values = Y_df['unique_id'].unique()[:series_limit]
    Y_df_5000 = Y_df[Y_df['unique_id'].isin(unique_values)]
    Y_df_test = Y_df_5000.groupby('unique_id').tail(group.horizon).copy()
    Y_df_train = Y_df_5000.drop(Y_df_test.index)
    Y_df_train["ds"] = Y_df_train["ds"].astype("int")
    Y_df_test["ds"] = Y_df_test["ds"].astype("int")

    list_models = ["DeepAR", "TCN", "NHITS", "MLP"]

    mae = {
        "unique_id": list(),
        "TCN": list(),
        "MLP": list(),
        "DeepAR": list(),
        "NHITS": list()
    }

    for i in unique_values:
        mae["unique_id"].append(i)
        for j in list_models:
            forecasts = pd.read_csv(f"./data/forecast/{j}.csv")
            forecasts_id = forecasts[forecasts['unique_id'] == i][j].values
            test = Y_df_test[Y_df_test['unique_id'] == i]["y"].values
            mae[j].append(mean_absolute_error(test, forecasts_id))
    
    pd.DataFrame(mae).to_csv("./data/base_performance/mae.csv", index=None)


