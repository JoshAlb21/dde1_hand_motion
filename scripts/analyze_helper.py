import pandas as pd

def check_for_missing_vals(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    missing_in_col = df.columns[df.isnull().any()]
    missing_in_col = missing_in_col.values.tolist()

    return missing_in_col

def compute_missing_ratio(df: pd.core.frame.DataFrame) -> pd.core.series.Series:

    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_value_df = missing_value_df.apply(lambda entry: round(entry, 2))
    missing_value_s = missing_value_df['percent_missing'].squeeze()
    missing_value_s = missing_value_s[missing_value_s != 0.0]

    return missing_value_s