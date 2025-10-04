import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def egfr(creat: float, age: float, gender: int, ethnicity: float) -> float:
    """
    Calculate the estimated glomerular filtration rate (eGFR).

    Inputs:
        - creat (float): Creatinine level.
        - age (float): Age in years.
        - gender (int): Gender.
        - ethnicity (float): Ethnicity.

    Outputs:
        - float or None: The calculated eGFR value or none.
    """
    # Return None if inputs are not correct
    if pd.isnull(creat) or creat == 0.0 or age == 0.0:
        return None  # type: ignore[arg-type]

    # Calculate eGFR
    factor_gender = 0.742 if gender == 0 else 1
    factor_ethnicity = 1.212 if ethnicity == 2 else 1
    eGFR = 175 * (creat**-1.154) * (age**-0.203) * factor_gender * factor_ethnicity

    return eGFR


def filter_missing(df: pd.DataFrame, th_col: float, th_row: float) -> pd.DataFrame:
    """
    Filters a DataFrame by removing columns with many missing values and rows
    with significant missing data.

    Inputs:
        - df (pd.DataFrame): The input DataFrame.
        - col_th(float): The maximum allowable number of NaN values per column.
        - row_th(float): The maximum allowable number of NaN values per row.

    Outputs:
        - pd.DataFrame: A filtered DataFrame.
    """
    # Filter columns based on NaN percentage threshold
    col_missing = df.isna().mean(axis=0)
    df_filt1 = df.loc[:, col_missing < th_col]

    # Filter rows based on NaN percentage threshold
    row_missing = df_filt1.isna().mean(axis=1)
    df_filt2 = df_filt1[row_missing < th_row]

    return df_filt2


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("../../data/raw/cohort.csv")

    # Map gender: 1 for male, 0 for female
    df["GENDER"] = df["GENDER"].replace({"M": 1, "F": 0}).astype(int)

    # Get the top 2 and top 1 ethnicities by frequency
    ethnicity_counts = df["ETHNICITY"].value_counts()
    top1 = ethnicity_counts.idxmax()
    top2 = ethnicity_counts.nlargest(2).index[1]

    # Map ethnicities: 1 for top 1, 2 for top 2, 3 for the rest
    df["ETHNICITY"] = df["ETHNICITY"].apply(
        lambda x: 1 if x == top1 else 2 if x == top2 else 3
    )

    # Manage datatypes
    for col in df.columns:
        if col not in ["GENDER", "ETHNICITY", "DM2", "CAD", "CKD", "HYP", "AKI"]:
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)

    # Compute 'MAX_EGFR'
    df["MAX_EGFR"] = df.apply(
        lambda row: egfr(
            row["MAX_CREATININE"], row["AGE_YEARS"], row["GENDER"], row["ETHNICITY"]
        ),
        axis=1,
    )

    # Compute 'MIN_EGFR'
    df["MIN_EGFR"] = df.apply(
        lambda row: egfr(
            row["MIN_CREATININE"], row["AGE_YEARS"], row["GENDER"], row["ETHNICITY"]
        ),
        axis=1,
    )

    # Drop columns 'SUBJECT_ID' and 'HADM_ID'
    df = df.drop(columns=["SUBJECT_ID", "HADM_ID"])

    # Drop columns 'MECHANICAL_VENTILATION' and 'VASOPRESSOR'
    df = df.drop(columns=["MECHANICAL_VENTILATION", "VASOPRESSOR"])

    # Filter out columns and rows with a lot of missing information
    df_clean = filter_missing(df, 0.15, 0.20)

    # Apply Iterative Imputer
    imputer = IterativeImputer(max_iter=10, random_state=123)
    df_impute = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

    # Separate features and target
    X = df_impute.drop(columns=["AKI"])
    y = df_impute["AKI"]

    # Identify binary columns
    binary_columns = ["GENDER", "DM2", "CAD", "CKD", "HYP"]

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=123, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # type: ignore[arg-type]

    # Post-process binary columns to ensure they remain 0 or 1
    for col in binary_columns:
        X_resampled[col] = (X_resampled[col] > 0.5).astype(int)

    # Post-process 'ETHNICITY' to ensure it remains 1, 2, or 3
    X_resampled["ETHNICITY"] = X_resampled["ETHNICITY"].round().clip(1, 3).astype(int)

    # Convert to DataFrame
    df_smote = pd.DataFrame(X_resampled, columns=X.columns)
    df_smote["AKI"] = y_resampled

    # Exploratory analysis
    print(df_smote.head())
    print(df_smote.info())
    print(df_smote.describe())

    # Define datasets and their names
    datasets = {
        "df": df,
        "df_clean": df_clean,
        "df_impute": df_impute,
        "df_smote": df_smote,
    }

    # Save NaN counts to a file
    os.makedirs("../../results/", exist_ok=True)
    with open("../../results/nan_counts.txt", "w", encoding="utf-8") as f:
        # Loop through each dataset
        for dataset_name, dataset in datasets.items():
            # Define masks for AKI and Non-AKI
            aki_mask = dataset["AKI"] == 1
            non_aki_mask = dataset["AKI"] == 0

            # Count NaN and non-NaN rows for AKI
            aki_nan_count = dataset.loc[aki_mask].isna().any(axis=1).sum()
            aki_non_nan_count = len(dataset.loc[aki_mask]) - aki_nan_count

            # Count NaN and non-NaN rows for Non-AKI
            non_aki_nan_count = dataset.loc[non_aki_mask].isna().any(axis=1).sum()
            non_aki_non_nan_count = len(dataset.loc[non_aki_mask]) - non_aki_nan_count

            # Write results to the file
            f.write(f"Metrics for {dataset_name}:\n")
            f.write(f"AKI NaN Count: {aki_nan_count}\n")
            f.write(f"AKI Non-NaN Count: {aki_non_nan_count}\n")
            f.write(f"Non-AKI NaN Count: {non_aki_nan_count}\n")
            f.write(f"Non-AKI Non-NaN Count: {non_aki_non_nan_count}\n")
            f.write("-" * 40 + "\n")

    # Save data
    os.makedirs("../../data/processed/", exist_ok=True)
    df_smote.to_csv("../../data/processed/cohort_smote.csv", index=False)
