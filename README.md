# Automobile-Insurance-Claim-Prediction2

## Project Overview

This project involves building a machine learning pipeline to predict whether a claim in an automobile insurance dataset is fraudulent or not. The dataset is unbalanced, requiring additional preprocessing steps such as oversampling using SMOTE. Multiple machine learning models were implemented and evaluated, including a stacking ensemble model.

The best-performing model was **Random Forest**, achieving an average accuracy of **79%**.

## Dataset

The dataset used in this project contains details about automobile insurance claims, including:

- **Numerical features**: `vehicle_claim`, `total_claim_amount`, `property_claim`, `injury_claim`, `umbrella_limit`, etc.
- **Categorical features**: `insured_sex`, `insured_relationship`, `policy_state`, etc.
- **Target variable**: `fraud_reported` (values: `Y` or `N`)

### Key Dataset Details

1. **Unbalanced Classes**: The target variable (`fraud_reported`) is highly unbalanced, necessitating the use of SMOTE (Synthetic Minority Oversampling Technique).
2. **Missing Values**: Columns such as `collision_type`, `police_report_available`, and `property_damage` contain missing values, filled using the mode.

## Exploratory Data Analysis

1. **Fraud Distribution**:
   - Count of `fraud_reported` (Y/N) visualized using a countplot.
2. **Feature Relationships**:
   - Pie charts for categorical distributions (`insured_sex`, `insured_relationship`).
   - Correlation heatmap of numerical features to identify relevant predictors for fraud.

## Preprocessing Steps

1. Replace missing values with the most frequent category (mode).
2. Apply SMOTE to balance the dataset.
3. Split the dataset into training (70%) and testing (30%) sets.

## Models Implemented

1. **Decision Tree** (DT)
2. **Random Forest** (RF)
3. **K-Nearest Neighbors** (KNN)
4. **XGBoost** (XGB)
5. **Stacking Ensemble**

### Stacking Model Architecture

- **Base Models**:
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - XGBoost
- **Meta Model**:
  - Logistic Regression

## Model Evaluation

Cross-validation was performed using 10-fold stratified splitting repeated 3 times.

### Results

| Model               | Mean Accuracy | Std. Deviation |
| ------------------- | ------------- | -------------- |
| Decision Tree       | 69%           | 4%             |
| Random Forest       | 79%           | 4%             |
| K-Nearest Neighbors | 68%           | 4%             |

**Best Model**: Random Forest (Accuracy = 79%)

## Usage Instructions

1. Clone the repository and place the dataset (`insurance_claims.csv`) in the appropriate folder.
2. Install required libraries:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn xgboost imbalanced-learn
   ```
3. Run the code to preprocess the data and train models:
   ```bash
   python main.py
   ```
4. The script outputs cross-validation results for each model and identifies the best-performing one.

## File Descriptions

- **main.py**: Contains the full pipeline from data preprocessing to model evaluation.
- **insurance\_claims.csv**: Dataset used for training and testing.
- **README.md**: Project documentation.
