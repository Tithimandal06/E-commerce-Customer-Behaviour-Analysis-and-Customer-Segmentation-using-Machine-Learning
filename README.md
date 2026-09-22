# E-Commerce Customer Behaviour Analysis and Customer Segmentation

**Author:** Tithi Mandal

## Project Overview
This research-oriented Python project analyzes 150 e-commerce customer records using the supplied customer-level dataset. It combines data quality checks, exploratory analysis, KPI reporting, RFM-style scoring, statistical tests, K-Means segmentation, PCA visualization, business interpretation, and an interactive Streamlit dashboard.

## Dataset
The seven source fields are `CustomerID`, `Age`, `Gender`, `PurchaseAmount`, `Frequency`, `Recency`, and `Category`. `CustomerID` is retained as an identifier. Age, purchase amount, frequency, and recency are the numerical behavioural features; gender and category are categorical features. The source contains 150 records, no missing values, no duplicate rows, and no duplicate customer IDs. Observed ranges are Age 20–55, PurchaseAmount 900–7300, Frequency 2–18, and Recency 2–84.

## Methodology
Data collection -> validation and cleaning -> EDA -> KPI and statistical analysis -> quantile-based RFM scores -> StandardScaler -> K-Means evaluation for K=2..8 -> silhouette-selected segmentation -> PCA visualization -> business insights.

RFM uses Recency, Frequency, and PurchaseAmount as the monetary proxy. Recency is scored in reverse because lower values indicate more recent activity. RFM segment thresholds are quantile-derived and documented in the generated report. K-Means uses Age, PurchaseAmount, Frequency, and Recency only; identifiers and categories are not treated as clustering features.

## Dashboard
`app.py` provides Executive Dashboard, Behaviour Analysis, RFM Analysis, Customer Segmentation, Customer Explorer, and Business Insights views. The sidebar supports CSV upload, filters, customer selection, and downloads. The app falls back to `customer_data.csv` when no file is uploaded and reports invalid schemas instead of crashing.

## Project Structure
```text
app.py
customer_data.csv
requirements.txt
README.md
generate_project.py
E-Commerce_Customer_Behaviour_Analysis_Report.docx
outputs/
  charts/
  customer_analysis.csv
  segmentation_results.csv
models/
  customer_segmentation_model.pkl
```

## Installation and Usage
```bash
python -m pip install -r requirements.txt
python generate_project.py
streamlit run app.py
```

The generation script creates the processed CSV, charts, fitted model, and the DOCX report from the supplied dataset. The included report contains actual calculated values and statistical results, not placeholders.

## Key Findings
The generated report is the source of truth for exact model-selected cluster names, scores, KPIs, tests, and category totals. The dashboard calculates these values at runtime so that uploaded datasets remain supported. Because the source is customer-level rather than transaction-level, recency is treated as a supplied behavioural measure; it is not converted into a calendar date.

## Limitations and Future Scope
The dataset has no timestamps, transaction lines, churn label, product attributes, or customer lifetime value. Therefore, the project supports descriptive segmentation rather than causal inference or churn prediction. Future work could add transaction-level CLV, recommendation systems, churn modelling, time-series analysis, real-time scoring, and larger datasets.