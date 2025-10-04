import pandas as pd
import pymysql
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from model.feature_selection import summary_table, vif_collinearity
from model.model_generator import logistic_model_evaluation
from processing.clean import egfr, filter_missing
from processing.normalize import normalize
from query.execute_query import execute_query

# Define connection parameters
HOST = ""
USER = ""
PASSWORD = ""
DATABASE = ""
PORT = 3306

# Connect to database
con = pymysql.connect(
    host=HOST,
    user=USER,
    password=PASSWORD,
    database=DATABASE,
    port=PORT,
)

# Read query
with open("query/cohort.sql", "r", encoding="utf-8") as file:
    query = file.read()

# Execute query
df = execute_query(con, query)

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

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=123, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)  # type: ignore[arg-type]

# Identify binary columns
binary_columns = ["GENDER", "DM2", "CAD", "CKD", "HYP"]

# Post-process binary columns to ensure they remain 0 or 1
for col in binary_columns:
    X_resampled[col] = (X_resampled[col] > 0.5).astype(int)

# Post-process 'ETHNICITY' to ensure it remains 1, 2, or 3
X_resampled["ETHNICITY"] = X_resampled["ETHNICITY"].round().clip(1, 3).astype(int)

# Convert to DataFrame
df_smote = pd.DataFrame(X_resampled, columns=X.columns)
df_smote["AKI"] = y_resampled

# Normalize DataFrame
df_norm = normalize(df_smote)

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

# Get the names of the selected features
selected_features = features_table["Feature"].tolist()
X = df_norm[selected_features]

# Evaluate model performance
metrics = logistic_model_evaluation(
    X,
    y,
    test_size=0.3,
    random_state=123,
    C=100,
    max_iter=200,
    penalty="l2",
)

# Print metrics
for set_name, metric in metrics.items():
    print(f"Metrics for {set_name.capitalize()} Set:\n")
    print(f"Confusion Matrix:\n{metric['conf_matrix']}\n")
    print(f"Accuracy: {metric['accuracy']:.4f}\n")
    print(f"Precision: {metric['precision']:.4f}\n")
    print(f"Recall: {metric['recall']:.4f}\n")
    print(f"F1 Score: {metric['f1']:.4f}\n")
    print(f"AUC: {metric['auc']:.4f}, 95% CI: {metric['ci']}\n")
    print("-" * 40 + "\n")
