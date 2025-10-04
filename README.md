<div align="justify">

# Acute Kidney Injury Prediction

This repository contains the complete workflow for modeling Acute Kidney Injury
(AKI) prediction in septic ICU patients using logistic regression. The project
includes code for data processing, model training, and visualization, as well
as LaTeX reports and presentation slides.

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

## 🎓 Acknowledgements

* Dataset: MIMIC-III clinical database
* Developed as part of the Health Data Science Master’s program at Universitat
Rovira i Virgili (URV)

</div>
