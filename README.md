<div align="justify">

# Acute Kidney Injury Prediction

This repository contains the complete workflow for modeling Acute Kidney Injury
(AKI) prediction in septic ICU patients using logistic regression. The project
includes code for data processing, model training, and visualization, as well
as LaTeX reports and presentation slides.

## 📝 Project

```
├── data/                   # Raw and processed datasets
│   ├── raw/                # Extracted data from MIMIC-III
│   └── processed/          # Cleaned dataset
├── images/                 # All generated visualizations
│   ├── coeffs.png          # Logistic regression coefficients
│   ├── egfr-function.png   # eGFR formula illustration
│   ├── errorbars.png       # AUC confidence intervals
│   ├── kidney-nephron.jpeg # Anatomical reference
│   ├── kidney-with-aki.jpg # Visual example of AKI
│   ├── logos/              # Logos for presentation/report
│   └── roc-curve.png       # ROC curves for training/testing
├── misc/                   # Additional files
│   └── refs.bib            # References for report
├── presentation/           # LaTeX files for the presentation
├── report/                 # LaTeX files for the project report
├── results/                # Output folders with metrics and summaries
│   ├── features.txt        # Final selected features
│   ├── metrics.txt         # Evaluation metrics
│   └── nan_counts.txt      # Missing value counts
├── src/                    # Source code for modeling
│   ├── main.py             # Entrypoint to run the full pipeline
│   ├── model/              # Logistic regression implementation
│   ├── processing/         # Preprocessing and feature selection
│   └── query/              # SQL-like utilities
├── LICENSE                 # License file
├── pyproject.toml          # Python project configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/RahulSandhu/aki-prediction
   cd aki-prediction
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 💻 Source

* `src/main.py`: Script to run preprocessing, feature selection, and modeling
* `src/model/`: Logistic regression model logic
* `src/processing/`: Data preprocessing, eGFR calculation, and feature
selection
* `src/query/`: Cohort building tools

## 📁 Data

* `data/raw/`: Raw MIMIC-III data extracted and restricted to the cohort
* `data/processed/`: Preprocessed dataset used for modeling

## 📊 Results

* `results/`: Metrics, selected features, missing value summaries
* Final model performance:
  * **AUC (Test Set)**: 0.810 (95% CI: 0.788–0.832)
  * **Accuracy**: 74.1%
  * **Precision**: 77.0%
  * **Recall**: 67.3%
  * **F1-score**: 71.8%
* Top predictors:
  * eGFR, Creatinine, BUN, Anion Gap, Potassium

## 📚 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## 🎓 Acknowledgements

* Dataset: MIMIC-III clinical database
* Developed as part of the Health Data Science Master’s program at Universitat
Rovira i Virgili (URV)

</div>
