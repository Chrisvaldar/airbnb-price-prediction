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

## Future Plans

**Possible model improvements**
- Log-transform the price target before training, since price is heavily right-skewed and the OOD RMSE being 2x+ the CV RMSE points to a generalisation gap this usually helps close
- Replace distance-to-Manhattan with a stronger feature, like distance to nearest subway station or borough-level price averages
- Compare price and neighbourhood distributions between the training and OOD datasets to actually diagnose what's shifting
- Swap manual/grid tuning for Optuna to get a more thorough hyperparameter search

**End-to-end app**
- Build a simple Streamlit app where you enter neighbourhood, room type, min nights, availability, review count, and lat/long, and get a predicted price back
- Load the saved XGBoost model in the backend for instant predictions
- Deploy free on Streamlit Community Cloud or HuggingFace Spaces so it's a live link, not just a notebook
## Data

- **Training:** [NYC Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) — `AB_NYC_2019.csv`
- **External test:** Airbnb listings dump (`listings.csv.gz`)
