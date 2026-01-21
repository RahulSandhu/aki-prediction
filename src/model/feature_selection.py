import os

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def vif_collinearity(df: pd.DataFrame, th: float = 10.0) -> list:
    """
    Calculate the Variance Inflation Factor (VIF) and iteratively remove
    features with VIF above the specified threshold.

    Inputs:
        - df (pd.DataFrame): Input DataFrame.
        - th (float): Threshold for VIF.

    Outputs:
        - list: List of non-collinear features.
    """
    # Iteratively remove collinear features
    while True:
        # Add a constant term to the DataFrame for VIF calculation
        df_with_const = add_constant(df)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame(
            {
                "Feature": df_with_const.columns,  # type: ignore[arg-type]
                "VIF": [
                    variance_inflation_factor(df_with_const.values, i)  # type: ignore[arg-type]
                    for i in range(df_with_const.shape[1])
                ],
            }
        )

        # Exclude the constant term from VIF evaluation
        vif_data = vif_data[vif_data["Feature"] != "const"]

        # Break the loop if all VIF values are below the threshold
        if vif_data["VIF"].max() < th:
            break

        # Identify the feature with the highest VIF
        feature_to_remove = vif_data.sort_values(by="VIF", ascending=False)[  # type: ignore[arg-type]
            "Feature"
        ].iloc[
            0
        ]

        # Remove the feature with the highest VIF from the DataFrame
        df = df.drop(columns=[feature_to_remove])

    return df.columns.tolist()


def summary_table(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Generate a summary table with mean, standard deviation, counts, and
    p-values.

    Inputs:
        - df (pd.DataFrame): Input DataFrame containing features.
        - non_collinear_features (list): List non-collinear features.
        - y (pd.Series): Target variable.

    Outputs:
        - pd.DataFrame: A DataFrame with statistics for quantitative and
          qualitative features.
    """
    # Define qualitative features
    qualitative = ["GENDER", "ETHNICITY", "DM2", "CAD", "CKD", "HYP"]
    feature_types = {
        "quantitative": [col for col in df.columns if col not in qualitative],
        "qualitative": [col for col in df.columns if col in qualitative],
    }

    # Split data into AKI and non-AKI
    aki_group = df[y == 1]
    non_aki_group = df[y == 0]

    # Preallocate array
    results = []

    # Process quantitative features
    for feature in feature_types["quantitative"]:
        aki_mean = aki_group[feature].mean()
        non_aki_mean = non_aki_group[feature].mean()
        aki_std = aki_group[feature].std()
        non_aki_std = non_aki_group[feature].std()

        # t-test
        _, p_value = ttest_ind(
            aki_group[feature], non_aki_group[feature], nan_policy="omit"
        )

        # Append results
        results.append(
            {
                "Feature": feature,
                "AKI Mean (SD)": f"{aki_mean:.3f} ({aki_std:.3f})",
                "Non-AKI Mean (SD)": f"{non_aki_mean:.3f} ({non_aki_std:.3f})",
                "P-Value": p_value,
            }
        )

    # Process qualitative features
    for feature in feature_types["qualitative"]:
        contingency_table = pd.crosstab(df[feature], y)

        # Chi-square test
        _, p_value, _, _ = chi2_contingency(contingency_table)

        # Append results
        results.append(
            {
                "Feature": feature,
                "AKI Count": aki_group[feature].value_counts().to_dict(),  # type: ignore[arg-type]
                "Non-AKI Count": non_aki_group[feature].value_counts().to_dict(),  # type: ignore[arg-type]
                "P-Value": p_value,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load dataset
    df_smote = pd.read_csv("../../data/processed/cohort_smote.csv")
    df_norm = pd.read_csv("../../data/processed/cohort_norm.csv")

    # Select numeric data and objective feature
    df_norm = df_norm.drop(columns=["AKI"])
    y = df_smote["AKI"]

    # Find non-collinear features
    non_collinear_features = vif_collinearity(df_norm, th=10)
    df_smote = df_smote[non_collinear_features]

    # Find the most significant features
    features_table = summary_table(df_smote, y)  # type: ignore[arg-type]

    # Filter by p-value
    features_table = features_table[features_table["P-Value"] <= 0.05]

    # Save metrics to a file
    os.makedirs("../../results/tables/", exist_ok=True)
    with open("../../results/tables/features.txt", "w", encoding="utf-8") as f:
        f.write(features_table.to_string(index=False))
