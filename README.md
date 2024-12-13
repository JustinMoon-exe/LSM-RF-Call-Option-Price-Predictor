# LSMRF-COPP: Enhancing American Call Option Pricing with Random Forests

This repository contains the code and documentation for the \textit{LSMRF-COPP} (Least Squares Monte Carlo - Random Forest - Call Option Price Predictor) model, a hybrid approach for pricing American call options.  The model combines the Least Squares Monte Carlo (LSM) algorithm with a Random Forest regressor to improve the accuracy of continuation value estimation and, consequently, option pricing.

This project was completed for GSU 4740 Data Mining

## Project Overview

Traditional LSM methods often rely on polynomial regression for estimating continuation values, which may not effectively capture the complex, non-linear relationships present in option pricing data.  \textit{LSMRF-COPP} addresses this limitation by integrating a Random Forest regressor, leveraging its ability to learn non-linear patterns and handle high-dimensional data.  The model is trained and evaluated using historical SPY option data from Yahoo Finance.

## Key Features

* **Hybrid Approach:** Combines LSM with Random Forest for enhanced accuracy.
* **Feature Engineering:** Incorporates a rich feature set, including option Greeks (Delta, Gamma, Theta, Vega), moneyness, time value, and combined features like Volatility $\times$ Time to Maturity.
* **Data Filtering and Preprocessing:**  Handles missing values, outliers, and ensures data quality for robust model training.
* **Hyperparameter Optimization:**  Uses a grid search with cross-validation to find optimal hyperparameters for the Random Forest model.
* **Comprehensive Evaluation:**  Evaluates the model's performance using key metrics such as MAPE, RMSE, and R-squared.
* **Visualization:**  Provides insightful visualizations of model performance, feature importance, and error distribution.

## Requirements

* Python 3.7+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* yfinance
* Scipy


## Results

The results demonstrate significant improvements in prediction accuracy compared to traditional LSM methods.  Refer to the paper [Link to your paper if available] for detailed results and analysis, including performance metrics, feature importance rankings, and visualizations.

## Acknowledgements

Special thanks to Professor Jingyu Liu for her guidance and support during the Data Mining class, which inspired this project.
