import os

import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns in the DataFrame by scaling values to the range [0, 1].

    Inputs:
        - df (pd.DataFrame): The input DataFrame.

    Outputs:
        - pd.DataFrame: A normalized DataFrame.
    """
    # Normalize columns using the Min-Max formula
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)  # type: ignore[arg-type]

    return df


if __name__ == "__main__":
    # Load dataset
    df_smote = pd.read_csv("../../data/processed/cohort_smote.csv")

    # Normalize DataFrame
    df_norm = normalize(df_smote)

    # Exploratory analysis
    print(df_norm.head())
    print(df_norm.info())
    print(df_norm.describe())

    # Save data
    os.makedirs("../../data/processed/", exist_ok=True)
    df_norm.to_csv("../../data/processed/cohort_norm.csv", index=False)
