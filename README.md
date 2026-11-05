# Predicting Severe Injury in Pedestrian and Cyclist Crashes and the Role of Vehicle Type

## Overview
This project uses the NHTSA Crash Report Sampling System (CRSS) dataset (2016–2023) to train supervised machine learning models (Random Forest, XGBoost) that predict whether a pedestrian or cyclist hit by a vehicle will sustain severe or fatal injuries.

## Key Objectives
- **Prediction:** Classify crash severity (Binary: Severe/Fatal vs. Non-Severe) using pre-injury crash parameters.
- **Temporal Validation:** Train on 2016–2021, validate on 2022, and evaluate on unseen 2023 crash data.
- **Feature Impact:** Utilize SHAP values and "what-if" counterfactual analysis to evaluate how vehicle type (cars vs. SUVs/trucks) affects predicted injury severity.

## Tech Stack
`Python` | `pandas` | `scikit-learn` | `XGBoost` | `SHAP`
