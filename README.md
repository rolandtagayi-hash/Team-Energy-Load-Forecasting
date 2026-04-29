# Robust Evaluation of Machine Learning Models for Building Energy Consumption Prediction Under Heavy-Tailed Meter Data

## Project Overview

This repository contains the code, paper materials, figures, and supporting files for a machine learning study on **building energy consumption prediction** using the **ASHRAE Great Energy Predictor III** dataset. The project evaluates how different regression models perform when predicting large-scale building meter readings that are highly skewed, heterogeneous, and affected by building type, meter category, site location, weather, and time-dependent operating patterns.

Unlike a conventional short-term load forecasting system that depends on autoregressive lag variables and fixed forecast horizons, this project treats the task as a **time-aware supervised regression problem**. The goal is to predict future-period hourly meter readings using available building descriptors, meter identifiers, site information, weather variables, and calendar-based features.

The central challenge addressed in this work is that real building meter data are **heavy-tailed**. Most observations represent normal building operation, while a smaller number of extreme readings dominate raw squared-error metrics. To handle this, the target variable is transformed using:

```text
z = log(1 + meter_reading)
```

Models are trained and evaluated on the log-transformed target and then converted back to the original meter-reading scale for practical interpretation. This allows the study to compare model behavior on both the logarithmic scale and the recovered natural scale.

---

## Why This Project Matters

Accurate building energy prediction supports:

- Energy analytics and benchmarking
- Measurement and verification
- Demand-side management
- Building operation planning
- Fault and anomaly screening
- Energy-efficiency assessment
- Data-driven smart building management

As buildings become more instrumented through smart meters, weather stations, automation systems, and energy management platforms, machine learning can help identify operating patterns and predict consumption. However, reliable model evaluation requires careful treatment of skewed targets, missing data, site differences, meter-type differences, and chronological validation.

This project focuses not only on which model gives the lowest error, but also on **why the error behaves differently across log-scale and original-scale metrics**.

---

## Main Research Questions

This study is guided by the following questions:

1. Do nonlinear ensemble models outperform linear and regularized linear baselines for hourly building meter prediction?
2. Does a log-transformed target provide a more stable modeling approach for strongly right-skewed meter readings?
3. How do natural-scale metrics and log-scale metrics differ when the target distribution contains rare extreme values?
4. What do actual-versus-predicted diagnostic plots reveal about model behavior under heavy-tailed building energy data?
5. What limitations remain before this framework can become a complete operational forecasting system?

---

## Key Contributions

This project provides:

1. A unified machine learning pipeline for large-scale building meter-reading prediction.
2. A fair comparison of Linear Regression, Ridge Regression, Lasso Regression, XGBoost, and LightGBM under the same preprocessing and validation design.
3. A log-target modeling strategy for reducing the effect of heavy right skew in meter readings.
4. Chronological validation to avoid future-period leakage.
5. Evaluation on both original and logarithmic target scales.
6. Diagnostic interpretation using actual-versus-predicted plots.
7. Evidence that boosted-tree models capture nonlinear and interaction-dependent energy-use patterns better than linear baselines.
8. A clear discussion of why rare high-consumption observations still require future tail-aware modeling.

---

## Dataset

The project uses the **ASHRAE Great Energy Predictor III** dataset, which contains hourly building energy meter data, building metadata, and weather information.

### Main Data Files

| File | Description |
|---|---|
| `train.csv` | Hourly meter readings. The target variable is `meter_reading`. |
| `building_metadata.csv` | Building-level information such as site, primary use, square footage, year built, and floor count. |
| `weather_train.csv` | Hourly weather measurements for each site. |

### Dataset Summary

| Item | Value |
|---|---:|
| Raw merged observations | 20,216,100 |
| Buildings | 1,449 |
| Sites | 16 |
| Observation period | January 1–December 31, 2016 |
| Temporal resolution | Hourly |
| Target variable | `meter_reading` |
| Transformed target | `log(1 + meter_reading)` |
| Training period | January–August 2016 |
| Validation period | September–December 2016 |
| Training observations after filtering | 12,675,376 |
| Validation observations after filtering | 6,685,946 |

### Meter Categories

The target variable includes multiple meter types. Because these meters represent different physical quantities and scales, original-scale errors should be interpreted carefully.

| Meter Type | Meter ID | Total Readings | Median Reading |
|---|---:|---:|---:|
| Electricity | 0 | 12,060,910 | 62.84375 |
| Chilled water | 1 | 4,182,440 | 120.50000 |
| Steam | 2 | 2,708,713 | 257.75000 |
| Hot water | 3 | 1,264,037 | 39.62500 |

---

## Repository Structure

A recommended repository organization is shown below.

```text
Team-Energy-Load-Forecasting/
│
├── README.md
├── Final Ashrae 1_2.ipynb
├── Team_Energy_Load_Forecasting.pdf
│
├── data/
│   ├── train.csv
│   ├── building_metadata.csv
│   └── weather_train.csv
│
├── figures/
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── temporal_patterns.png
│   ├── workflow_diagram.png
│   ├── natural_scale_metrics.png
│   ├── log_scale_metrics.png
│   └── actual_vs_predicted.png
│
├── outputs/
│   ├── model_results.csv
│   ├── trained_models/
│   └── diagnostic_plots/
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
└── requirements.txt
```

The exact folder names may be adjusted, but separating raw data, figures, outputs, notebook files, and paper files will make the project easier to reproduce and review.

---

## Methodological Workflow

The complete workflow follows these stages:

1. Load ASHRAE meter, building metadata, and weather data.
2. Merge the data using `building_id`, `site_id`, and `timestamp`.
3. Remove known data-quality problems and nonfinite values.
4. Engineer building, meter, weather, and calendar features.
5. Apply a log transformation to the target variable.
6. Split the data chronologically into training and validation sets.
7. Train five regression models under the same evaluation framework.
8. Recover predictions back to the original meter-reading scale.
9. Evaluate each model using natural-scale and log-scale metrics.
10. Generate diagnostic plots and compare model behavior.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Merging meter data with building metadata and weather observations
- Parsing timestamps into calendar features
- Removing nonfinite target values
- Removing selected unreliable building identifiers
- Correcting the known early electricity-meter zero anomaly before May 20, 2016
- Handling missing values in weather and metadata fields
- Encoding categorical variables such as `primary_use`
- Applying memory-aware data handling for the large dataset
- Transforming the response variable using `log1p(meter_reading)`

The log transformation is important because the original `meter_reading` distribution is extremely right-skewed. Training directly on the original target would allow rare extreme readings to dominate squared-error behavior.

---

## Feature Engineering

The final feature set includes several groups of predictors.

### Building and Site Features

- `building_id`
- `site_id`
- `meter`
- `primary_use`
- `square_feet`
- `log_square_feet`
- `year_built`, where available
- `floor_count`, where available

### Weather Features

- `air_temperature`
- `dew_temperature`
- `cloud_coverage`
- `precip_depth_1_hr`
- `sea_level_pressure`
- `wind_direction`
- `wind_speed`

### Calendar Features

- Hour of day
- Day of week
- Day of month
- Month
- Weekend indicator
- Timestamp-based chronological ordering

These features are used because building energy consumption depends on time of day, weekly schedules, building function, site climate, meter category, and weather conditions.

---

## Models Evaluated

The study compares five regression models.

| Model | Purpose in the Study |
|---|---|
| Linear Regression | Transparent baseline for global linear structure. |
| Ridge Regression | Regularized linear baseline that handles correlated predictors using an L2 penalty. |
| Lasso Regression | Sparse linear baseline using an L1 penalty. |
| XGBoost | Nonlinear boosted-tree model designed to capture feature interactions and complex regimes. |
| LightGBM | Efficient boosted-tree model suitable for large datasets and nonlinear interactions. |

The linear models establish whether the selected predictors contain useful global additive structure. The boosted-tree models test whether nonlinear interactions among building, meter, weather, site, and time features improve predictive accuracy.

---

## Validation Design

A chronological split is used instead of a random split.

| Split | Period |
|---|---|
| Training | January–August 2016 |
| Validation | September–December 2016 |

This prevents future observations from leaking into the training set and better represents real future-period prediction. Chronological validation is especially important for energy data because consumption patterns can vary with season, weather, occupancy, and operating schedules.

The study also considers expanding-window validation blocks during model development, where each fold trains on earlier observations and validates on a later time block.

---

## Evaluation Metrics

The models are evaluated on both the original meter-reading scale and the log-transformed scale.

### Original-Scale Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (`R²`)

### Log-Scale Metrics

- Root Mean Squared Logarithmic Error (RMSLE)
- Log-scale coefficient of determination (`R²_log`)

RMSLE is equivalent to RMSE on the transformed target `log(1 + meter_reading)`. This metric is useful because it penalizes approximate multiplicative error and is less dominated by rare extreme readings than original-scale RMSE.

---

## Final Validation Results

The table below summarizes the validation performance on the September–December 2016 period.

| Model | RMSE | RMSLE | MAE | R² | R²_log |
|---|---:|---:|---:|---:|---:|
| XGBoost | 43,105.5 | 1.2588 | 487.6 | 0.0051 | 0.6364 |
| LightGBM | 43,118.7 | 1.2628 | 464.4 | 0.0045 | 0.6341 |
| Linear Regression | 43,217.2 | 1.8282 | 570.1 | -0.0001 | 0.2330 |
| Ridge Regression | 43,218.9 | 2.3453 | 613.2 | -0.0002 | -0.2622 |
| Lasso Regression | 43,218.9 | 2.3413 | 612.8 | -0.0002 | -0.2579 |

### Main Result

XGBoost gives the strongest overall performance based on RMSE, RMSLE, and log-scale explained variance. LightGBM performs very closely and gives the lowest MAE. The linear models provide useful baselines but do not capture the nonlinear and interaction-dependent structure of the building energy data as effectively as the boosted-tree models.

---

## Interpretation of Results

The results show that target scale strongly affects interpretation.

On the log scale, XGBoost and LightGBM clearly outperform the linear-family models. This means the boosted-tree models successfully learn important ordinary operating patterns in the data.

On the original meter-reading scale, all models have natural-scale `R²` values close to zero. This does not mean the models learn nothing. Instead, it shows that rare extreme readings dominate raw squared-error behavior. A model can improve millions of ordinary predictions while still appearing weak under original-scale `R²` if it misses a small number of very large observations.

Therefore, the best conclusion is not simply that one model wins. The stronger conclusion is that **log-scale boosted-tree modeling is effective for ordinary building energy prediction, but rare extreme readings require a future tail-aware modeling layer**.

---

## Figures and Diagnostics

The paper and notebook include several important visual diagnostics:

1. Raw and log-transformed target distributions
2. Correlation matrix of selected numerical variables
3. Average load patterns by hour, day of week, and annual timeline
4. Research workflow diagram
5. Natural-scale RMSE and MAE comparison
6. RMSLE and `R²_log` comparison
7. Natural-scale `R²` comparison
8. Actual-versus-predicted plots on the original scale
9. Actual-versus-predicted plots on logarithmic axes

These figures support the main finding that nonlinear boosted-tree models capture the central log-scale structure better than linear baselines, while the largest readings remain difficult to predict accurately.

---

## How to Run the Project

### Option 1: Run in Google Colab

1. Open the project notebook:

```text
Final Ashrae 1_2.ipynb
```

2. Upload the ASHRAE dataset files or mount Google Drive.
3. Make sure the following files are available in the expected data directory:

```text
train.csv
building_metadata.csv
weather_train.csv
```

4. Install any missing packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm hyperopt joblib
```

5. Run the notebook cells in order.
6. Review the generated model metrics, plots, and saved outputs.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/estheromoyiwola/Team-Energy-Load-Forecasting.git
cd Team-Energy-Load-Forecasting
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset files in the `data/` folder, then open the notebook:

```bash
jupyter notebook "Final Ashrae 1_2.ipynb"
```

Run the cells sequentially from data loading through final evaluation.

---

## Suggested `requirements.txt`

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
hyperopt
joblib
jupyter
```

Depending on the local system, XGBoost and LightGBM installation may require compatible compiler or binary-wheel support.

---

## Expected Outputs

Running the notebook should produce:

- Cleaned and merged modeling data
- Exploratory plots
- Model training outputs
- Cross-validation or validation metrics
- Final comparison table
- RMSE, MAE, RMSLE, `R²`, and `R²_log` plots
- Actual-versus-predicted diagnostic plots
- Saved model result files, depending on the notebook configuration

---

## Limitations

This project has several important limitations:

1. It is a time-aware supervised prediction study, not a complete autoregressive short-term forecasting system.
2. The current model does not include lagged meter-reading features.
3. Original-scale metrics combine different meter types, so meter-specific evaluation is needed for stronger physical interpretation.
4. Rare extreme readings remain difficult to predict accurately.
5. The current framework does not yet include uncertainty intervals.
6. Explainability methods such as SHAP or permutation importance are not yet fully integrated.
7. Weather variables are historical observations; an operational forecasting system would require weather forecasts.

---

## Future Work

Future extensions should include:

- Lagged meter-reading features
- Horizon-specific forecasting targets
- Meter-specific or building-cluster models
- Weather-ablation experiments
- Robust loss functions for extreme values
- Quantile regression or prediction intervals
- Tail-risk and anomaly-aware modeling
- SHAP or permutation-based interpretability
- Deployment-oriented evaluation using forecasted weather inputs
- Separate evaluation for electricity, chilled water, steam, and hot-water meters

---

## Team

**Team Name:** Team 6

| Name | Department / Role |
|---|---|
| Esther Omoyiwola | Electrical and Computer Engineering; Team Lead |
| Roland Kobla Tagayi | Electrical and Computer Engineering |
| Yvonne Okafor | Computer Science |
| Akinfewa Ayobami | Mechanical Engineering |

---

## Project Paper

The accompanying paper is titled:

```text
Robust Evaluation of Machine Learning Models for Building Energy Consumption Prediction Under Heavy-Tailed Meter Data
```

It provides the full research motivation, literature review, mathematical formulation, dataset description, model comparison, evaluation results, diagnostic plots, limitations, and future research directions.

---

## Acknowledgment

This project uses the ASHRAE Great Energy Predictor III dataset and builds on the broader research direction of data-driven building energy analytics. The project is intended for academic study, reproducible machine learning evaluation, and building energy prediction research.

---
