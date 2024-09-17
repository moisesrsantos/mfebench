import pandas as pd

def get_min_col_name(row):
    return row.idxmin()

if __name__ == "__main__":
    # metatarget
    performances = pd.read_csv("./data/base_performance/mae.csv", index_col=0)
    metatarget = performances.apply(get_min_col_name, axis=1)
    metatarget.name = "target"

    # meta-features
    catch22 = pd.read_csv("./data/mf/catch22.csv", index_col=0)
    catch22.index.name = "unique_id"
    tsfeatures = pd.read_csv("./data/mf/tsfeatures.csv", index_col="unique_id")
    tsfel = pd.read_csv("./data/mf/tsfel.csv", index_col=0)
    tsfel.index.name = "unique_id"
    tsfresh = pd.read_csv("./data/mf/tsfresh.csv", index_col=0)
    tsfresh.index.name = "unique_id"

    # metadata
    pd.merge(catch22, metatarget, on='unique_id', how='inner').to_csv("./data/metadata/catch22.csv")
    pd.merge(tsfeatures, metatarget, on='unique_id', how='inner').to_csv("./data/metadata/tsfeatures.csv")
    pd.merge(tsfel, metatarget, on='unique_id', how='inner').to_csv("./data/metadata/tsfel.csv")
    pd.merge(tsfresh, metatarget, on='unique_id', how='inner').to_csv("./data/metadata/tsfresh.csv")



