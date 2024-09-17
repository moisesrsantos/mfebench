import pandas as pd
from pycatch22 import catch22_all
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
import tsfel
from tsfeatures import tsfeatures
import glob
from datasetsforecast.m4 import M4, M4Info
import os

class MFE:
    def __init__(self, series: pd.DataFrame) -> None:
        self.series = series
    
    @property
    def catch22(self) -> pd.DataFrame:
        metafeatures = list()
        unique_values = self.series['unique_id'].unique()
        for i in range(len(unique_values)):
            if i == 0:
                metafeatures.append(catch22_all(self.series[self.series['unique_id'] == unique_values[i]]["y"],catch24=False)["names"])
                metafeatures.append(catch22_all(self.series[self.series['unique_id'] == unique_values[i]]["y"],catch24=False)["values"])
            else:
                metafeatures.append(catch22_all(self.series[self.series['unique_id'] == unique_values[i]]["y"],catch24=False)["values"])
        return pd.DataFrame(metafeatures[1:], columns=metafeatures[0], index=unique_values)
    
    @property
    def tsfresh(self) -> pd.DataFrame:
        metafeatures = list()
        mfe_tsfresh = TSFreshFeatureExtractor(default_fc_parameters="comprehensive", show_warnings=False)
        unique_values = self.series['unique_id'].unique()
        for i in range(len(unique_values)):
            if i == 0:
                mf = mfe_tsfresh.fit_transform(self.series[self.series['unique_id'] == unique_values[i]]["y"])
                metafeatures.append(mf.columns.values.tolist())
                metafeatures.append(mf.values.tolist()[0])
            else:
                mf = mfe_tsfresh.fit_transform(self.series[self.series['unique_id'] == unique_values[i]]["y"])
                metafeatures.append(mf.values.tolist()[0])
        return pd.DataFrame(metafeatures[1:], columns=metafeatures[0], index=unique_values)
    
    def _read_and_prepare_csv(self, file_path):
        return pd.read_csv(file_path)
    
    @property
    def tsfel(self) -> pd.DataFrame:
        metafeatures = list()
        unique_values = self.series['unique_id'].unique()
        cfg = tsfel.get_features_by_domain()
        for i in range(len(unique_values)):
            mf = tsfel.time_series_features_extractor(cfg, self.series[self.series['unique_id'] == unique_values[i]]["y"].reset_index(drop=True), header_names = None, verbose = 0)
            mf.to_csv(f"./data/mf/tsfel_temp/{unique_values[i]}.csv", index=unique_values[i])
        file_paths = glob.glob("./data/mf/tsfel_temp/*.csv")
        file_list = [os.path.basename(file).strip(".csv") for file in file_paths]
        for file_path in file_paths:
            df = self._read_and_prepare_csv(file_path)
            metafeatures.append(df)
        merged_df = pd.concat(metafeatures, sort=False)
        merged_df.index = file_list
        df_cleaned = merged_df.dropna(axis=1)
        df_cleaned = df_cleaned.drop(["Unnamed: 0"], axis=1)
        df_cleaned.columns = df_cleaned.columns.str.lstrip("0_")
        return df_cleaned

    @property
    def tsfeatures(self) -> pd.DataFrame:
        return tsfeatures(self.series, freq=12)


if __name__ == "__main__":
    group = M4Info['Monthly']
    Y_df, _, S_df = M4.load(directory='data', group=group.name)

    series_limit = 5000

    unique_values = Y_df['unique_id'].unique()[:series_limit]
    Y_df_5000 = Y_df[Y_df['unique_id'].isin(unique_values)]
    Y_df_test = Y_df_5000.groupby('unique_id').tail(group.horizon).copy()
    Y_df_train = Y_df_5000.drop(Y_df_test.index)
    Y_df_train["ds"] = Y_df_train["ds"].astype("int")

    MFE(series=Y_df_train).catch22.to_csv("./data/mf/catch22.csv")
    MFE(series=Y_df_train).tsfresh.to_csv("./data/mf/tsfresh.csv")
    MFE(series=Y_df_train).tsfel.to_csv("./data/mf/tsfel.csv")
    MFE(series=Y_df_train).tsfeatures.to_csv("./data/mf/tsfeatures.csv", index = None)

