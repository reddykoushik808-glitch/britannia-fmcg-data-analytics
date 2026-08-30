# Britannia FMCG Data Analytics & Sales Prediction

Python data cleaning pipeline and sales prediction modeling for Britannia FMCG transaction data.

> Note: Dataset is synthetic and used for FMCG analytics benchmarking.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── 01_raw_data.csv
│   ├── interim/
│   │   ├── 02_after_duplicate_check.csv
│   │   ├── 03_after_date_cleaning.csv
│   │   ├── 04_after_missing_value_treatment.csv
│   │   ├── 05_after_text_cleaning.csv
│   │   ├── 06_after_geography_validation.csv
│   │   ├── 07_after_numeric_cleaning.csv
│   │   └── 08_business_rule_validated.csv
│   └── processed/
│       ├── final_clean_britannia_dataset.csv
│       ├── engineered_features.csv
│       └── scaled_features.csv
├── models/
│   ├── regression_model.pkl
│   └── preprocessing_pipeline.pkl
├── reports/
│   ├── missing_value_report.csv
│   ├── geography_validation_report.csv
│   ├── numeric_quality_report.csv
│   ├── business_rule_validation_report.csv
│   ├── model_comparison.csv
│   ├── regression_coefficients.csv
│   ├── regression_equation.txt
│   └── future_sales_predictions.csv
├── src/
│   ├── 01_profile_raw_data.py
│   ├── 02_remove_duplicates.py
│   ├── 03_clean_dates.py
│   ├── 04_handle_missing_values.py
│   ├── 05_clean_text.py
│   ├── 06_validate_geography.py
│   ├── 07_clean_numeric_data.py
│   ├── 08_validate_business_rules.py
│   ├── 09_feature_engineering.py
│   ├── 10_feature_scaling.py
│   ├── 11_train_regression.py
│   ├── 12_generate_regression_equation.py
│   ├── 13_predict_future_sales.py
│   └── generate_readme.py
├── README.md
└── requirements.txt
```

## Data Cleaning Stages

| Stage | Scope | Issue Identified | Action / Method | Output |
| :--- | :--- | :--- | :--- | :--- |
| **01** | Raw Profiling | 10,250 rows, 52 columns | Preserved raw dataset copy | `01_raw_data.csv` |
| **02** | Duplicates | 250 exact duplicate rows | `drop_duplicates()` (10,000 rows remaining) | `02_after_duplicate_check.csv` |
| **03** | Date Cleaning | 25 invalid/missing dates | Sequence forward/backward fill | `03_after_date_cleaning.csv` |
| **04** | Missing Values | Nulls in price, qty, ratings | Product-group median & category mode | `04_after_missing_value_treatment.csv` |
| **05** | Text Normalization | Formatting & typos (`Snaks`, `Delivred`) | Trimmed whitespace & string replace | `05_after_text_cleaning.csv` |
| **06** | Geography Check | 179 State-City mismatches | Corrected State via City lookup table | `06_after_geography_validation.csv` |
| **07** | Numeric Cleaning | Negative prices, Qty=9999 | Bounded outliers & clipped ratings | `07_after_numeric_cleaning.csv` |
| **08** | Business Rules | Calculated vs stored math errors | Recomputed Gross/Net Sales, COGS & Margin | `final_clean_britannia_dataset.csv` |

## Model Performance

Time-based split (Train: Aug 2023–Dec 2025 | Test: Jan 2026–Aug 2026):

| Model | MAE (Rs.) | RMSE (Rs.) | R² | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Multiple Linear Regression | 348.19 | 474.59 | 0.8759 | 130.47% |
| Ridge Regression | 348.17 | 474.61 | 0.8759 | 130.59% |
| Random Forest Regressor | 37.18 | 58.64 | 0.9981 | 3.08% |
| Gradient Boosting Regressor | 46.64 | 60.23 | 0.9980 | 9.86% |

## Linear Regression Equation

```text
Predicted_Net_Sales = 1583.53 + 913.53*(Quantity) + 847.94*(Unit_Price) - 105.90*(Discount_Pct) - 21.26*(Unit_Cost) + ...
```

- Equation details: `reports/regression_equation.txt`
- Coefficients table: `reports/regression_coefficients.csv`

## How to Run

Run the scripts in order:

```bash
pip install -r requirements.txt
python src/01_profile_raw_data.py
python src/02_remove_duplicates.py
python src/03_clean_dates.py
python src/04_handle_missing_values.py
python src/05_clean_text.py
python src/06_validate_geography.py
python src/07_clean_numeric_data.py
python src/08_validate_business_rules.py
python src/09_feature_engineering.py
python src/10_feature_scaling.py
python src/11_train_regression.py
python src/12_generate_regression_equation.py
python src/13_predict_future_sales.py
```
