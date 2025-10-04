import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from model.feature_selection import summary_table, vif_collinearity


def compute_auc_ci(y_true, y_proba, n_bootstraps=1000, random_state=123) -> tuple:
    """
    Computes the AUC (Area Under the Curve) and its 95% confidence interval
    using bootstrapping.

    Inputs:
        - y_true (array): True binary labels.
        - y_proba (array): Predicted probabilities for the positive class.
        - n_bootstraps (int): Number of bootstrap samples to generate.
        - random_state (int): Seed for random number generator.

    Outputs:
        - tuple:
            - auc_score (float): AUC score for the given true and predicted
              labels.
            - ci (tuple): A tuple containing the lower and upper bounds of the
              95% confidence interval.
    """
    # Initialize random state for reproducibility
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []

    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_proba)

    # Perform bootstrapping
    for _ in range(n_bootstraps):
        # Randomly sample indices with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)

        # Ensure the sampled data contains both classes
        if len(np.unique(y_true[indices])) < 2:
            continue

        # Compute ROC curve and AUC for the sampled data
        fpr, tpr, _ = roc_curve(y_true[indices], y_proba[indices])
        bootstrapped_scores.append(auc(fpr, tpr))

    # Compute the confidence interval
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)

    # Compute AUC for the original data
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    ci = (lower, upper)

    # ROC curve
    plt.plot(fpr, tpr)

    return auc_score, ci


def logistic_model_evaluation(
    X,
    y,
    test_size=0.3,
    random_state=123,
    C=100,
    max_iter=200,
    penalty="l2",
) -> dict:
    """
    Train a model and evaluate its performance on training and testing sets.

    Inputs:
        - X (pd.DataFrame): Feature set.
        - y (pd.Series): Target variable.
        - test_size (float): Proportion of the data to include in the test
          split.
        - random_state (int): Random seed for reproducibility.
        - C (float): Inverse of regularization strength for Logistic
          Regression.
        - max_iter (int): Maximum number of iterations for solver convergence.
        - penalty (str): Norm used in penalization.

    Outputs:
        - dict: Metrics and performance for train and test sets.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train the model
    lr_model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty)
    lr_model.fit(X_train, y_train)

    # Predictions
    predictions = {
        "train": (
            y_train,
            lr_model.predict(X_train),
            lr_model.predict_proba(X_train)[:, 1],
        ),
        "test": (
            y_test,
            lr_model.predict(X_test),
            lr_model.predict_proba(X_test)[:, 1],
        ),
    }

    # Preallocate dictionary
    results = {}

    # Preallocate array
    lgd_labels = []

    # Plot 1: ROC curve

    # Create figure
    plt.figure()

    # Reference line
    plt.plot([0, 1], [0, 1], "k--")
    lgd_labels.append("Reference")

    # Get metrics for training and testing sets
    for set_name, (y_set, y_pred, y_proba) in predictions.items():
        conf_matrix = confusion_matrix(y_set, y_pred)
        accuracy = accuracy_score(y_set, y_pred)
        precision = precision_score(y_set, y_pred)
        recall = recall_score(y_set, y_pred)
        f1 = f1_score(y_set, y_pred)

        # Compute AUC and CI
        auc_score, ci = compute_auc_ci(y_set, y_proba, random_state=random_state)

        # Define label
        label = f"{set_name.capitalize()} (AUC = {auc_score:.3f})"
        lgd_labels.append(label)

        # Update results
        results[set_name] = {
            "conf_matrix": conf_matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score,
            "ci": ci,
        }

    # Add legend
    plt.legend(lgd_labels, loc="lower right")

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Save plot
    os.makedirs("../../images/", exist_ok=True)
    plt.savefig(
        "../../images/roc_curve.png",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )

    # Close plot
    plt.close()

    # Plot 2: Coefficient contribution

    coeffs = abs(lr_model.coef_[0])
    x_pos = np.arange(len(coeffs))
    labels = X.columns.tolist()

    # Create figure
    plt.figure(figsize=(8, 6))

    # Bar chart
    plt.barh(x_pos, coeffs, color="skyblue", edgecolor="black")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Customize plot
    plt.yticks(x_pos, labels)
    plt.title("Feature Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Features")

    # Save plot
    os.makedirs("../../images/", exist_ok=True)
    plt.savefig(
        "../../images/coeffs.png",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )

    # Close plot
    plt.close()

    return results


if __name__ == "__main__":
    # Fix for Wayland
    matplotlib.use("QtAgg")

    # Use custom style
    plt.style.use("../../config/matplotlib/mhedas.mplstyle")

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

    # Get the names of the selected features
    selected_features = features_table["Feature"].tolist()
    X = df_norm[selected_features]

    # Evaluate model performance
    metrics = logistic_model_evaluation(X, y)

    # Save metrics to a file
    os.makedirs("../../results/", exist_ok=True)
    with open("../../results/metrics.txt", "w", encoding="utf-8") as f:
        for set_name, metric in metrics.items():
            f.write(f"Metrics for {set_name.capitalize()} Set:\n")
            f.write(f"Confusion Matrix:\n{metric['conf_matrix']}\n")
            f.write(f"Accuracy: {metric['accuracy']:.4f}\n")
            f.write(f"Precision: {metric['precision']:.4f}\n")
            f.write(f"Recall: {metric['recall']:.4f}\n")
            f.write(f"F1 Score: {metric['f1']:.4f}\n")
            f.write(f"AUC: {metric['auc']:.4f}, 95% CI: {metric['ci']}\n")
            f.write("-" * 40 + "\n")

    # Plot 3: Benchmarking against other studies

    # Select models
    models = ["Roknaldin et al.", "Malhotra et al.", "Jiang et al.", "Proposed Method"]

    auc_means = [0.887, 0.810, 0.760, 0.810]
    auc_lower = [0.861, 0.780, 0.700, 0.788]
    auc_upper = [0.915, 0.830, 0.820, 0.832]

    # Calculate error bars
    auc_errors_lower = np.array(auc_means) - np.array(auc_lower)
    auc_errors_upper = np.array(auc_upper) - np.array(auc_means)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Errorbar
    plt.errorbar(
        models,
        auc_means,
        yerr=[auc_errors_lower, auc_errors_upper],
        fmt="o",
        capsize=5,
        color="blue",
        ecolor="black",
        elinewidth=1.5,
        capthick=1.5,
    )

    # Customize plot
    plt.title("AUC with Confidence Intervals")
    plt.xlabel("Models")
    plt.ylabel("AUC")
    plt.ylim(0.6, 1.0)

    # Save plot
    os.makedirs("../../images/", exist_ok=True)
    plt.savefig(
        "../../images/errorbars.png", bbox_inches="tight", dpi=300, transparent=True
    )

    # Close plot
    plt.close()
