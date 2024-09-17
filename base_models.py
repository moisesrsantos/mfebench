import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, NHITS, TCN, MLP
from mlforecast.target_transforms import Differences
from datasetsforecast.m4 import M4, M4Info

class Base_Models:

    def __init__(self, train, h) -> None:
        self.train = train
        self.h = h

    def tcn(self):
        model = TCN(h=self.h, max_steps=1000, input_size=2 * self.h, scaler_type='robust', accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=12)
        nf.fit(df=self.train)
        return nf.predict()
        
    def deepar(self):
        model = DeepAR(h=self.h, max_steps=1000, input_size=2 * self.h, scaler_type='robust', accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=12)
        nf.fit(df=self.train)
        return nf.predict()
    
    def nhits(self):
        model = NHITS(h=self.h, max_steps=1000, input_size=2 * self.h, scaler_type='robust', accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=12)
        nf.fit(df=self.train)
        return nf.predict()

    def mlp(self):
        model = MLP(h=self.h, max_steps=1000, input_size=2 * self.h, scaler_type='robust', accelerator='cpu', enable_checkpointing=False, logger=False, random_seed=14)
        nf = NeuralForecast(models=[model], freq=12)
        nf.fit(df=self.train)
        return nf.predict()

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

    Base_Models(train=Y_df_train, h=group.horizon).mlp().to_csv("./data/forecast/MLP.csv")
    Base_Models(train=Y_df_train, h=group.horizon).tcn().to_csv("./data/forecast/TCN.csv")
    Base_Models(train=Y_df_train, h=group.horizon).nhits().to_csv("./data/forecast/NHITS.csv")
    Base_Models(train=Y_df_train, h=group.horizon).deepar().to_csv("./data/forecast/DeepAR.csv")