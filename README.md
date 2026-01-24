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
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset

The project analyzes data from the MIMIC-III (Medical Information Mart for
Intensive Care III) clinical database, focusing on septic ICU patients at risk
for Acute Kidney Injury. The dataset includes comprehensive clinical and
laboratory measurements with key features:

- **Renal function markers**: eGFR (estimated Glomerular Filtration Rate),
  creatinine, blood urea nitrogen (BUN)
- **Electrolyte measures**: Potassium, sodium, anion gap, bicarbonate
- **Vital signs**: Blood pressure, heart rate, temperature
- **Laboratory values**: White blood cell count, hemoglobin, platelets
- **Clinical indicators**: Sepsis severity scores, comorbidity indices
- **Outcome measure**: Development of Acute Kidney Injury (AKI)

The analysis uses **logistic regression** with feature selection to predict AKI
occurrence in the septic ICU patient population.

## 📊 Results

- Final model performance achieved **AUC of 0.810** (95% CI: 0.788–0.832) on
  the test set
- Model demonstrates robust classification with accuracy of **74.1%**,
  precision of **77.0%**, recall of **67.3%**, and F1-score of **71.8%**
- Top predictive features identified: eGFR, creatinine, BUN, anion gap, and
  potassium
- Feature selection and preprocessing pipeline optimized for clinical
  interpretability and prediction accuracy

## 🎓 Acknowledgements

- [PhysioNet MIMIC-III Clinical
  Database](https://physionet.org/content/mimiciii/1.4/) – Medical Information
  Mart for Intensive Care III
- Developed as part of the Electronic Health Records course in the Master in
  Health Data Science program at Universitat Rovira i Virgili (URV)

</div>
