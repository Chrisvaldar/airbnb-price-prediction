# Airbnb Price Prediction

Predict nightly prices for NYC Airbnb listings using an XGBoost regression model.

## Overview

Trained and tuned on **48,000+** NYC listings (`AB_NYC_2019`), then validated on a separate, out-of-distribution listings dataset to check real-world generalisation.

## Approach

- Clean and encode listing features (neighbourhood, room type, reviews, availability, etc.)
- Engineer a distance-to-Manhattan feature from latitude/longitude
- Train an `XGBRegressor` with tuned hyperparameters
- Evaluate with 5-fold cross-validation, then on the held-out external dataset

## Results

| Evaluation | RMSE |
|---|---|
| Cross-validation (in-distribution) | ~65 |
| Independent test set (OOD) | ~147 |

## Requirements

- Python 3
- pandas, numpy, matplotlib
- scikit-learn
- xgboost

## Usage

1. Place the datasets in a local `data/` folder (or update the paths in the notebook).
2. Open and run `AirBnB.ipynb`.

## Data

- **Training:** [NYC Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — `AB_NYC_2019.csv`
- **External test:** Airbnb listings dump (`listings.csv.gz`)
