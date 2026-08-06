Data Preprocessing, Feature Selection, and Model Training
========================================================

This project demonstrates a complete machine learning workflow using the Wine dataset.
It covers data preprocessing, missing value handling, categorical encoding, feature scaling,
feature selection, and model training.

What the code does
------------------
The script:
- Loads the Wine dataset from scikit-learn.
- Introduces missing values to show how missing data can be handled.
- Compares simple data-cleaning strategies such as dropping rows or columns and using mean imputation.
- Demonstrates categorical data encoding with OneHotEncoder and OrdinalEncoder.
- Applies normalization and standardization to the features.
- Uses Sequential Backward Selection (SBS) to reduce the number of features and improve model performance.
- Trains and evaluates a KNN classifier before and after feature selection.
- Uses Random Forest feature importance to rank the most useful predictors.
- Applies L1 regularized logistic regression to inspect feature coefficients.

Outputs
-------
The script generates several plots and saves them in the homework4plots folder, including:
- SBS accuracy plot
- Random Forest feature importance plot
- L1 regularization coefficient plot

How to run
----------
Run the script with:

python D-Preprocessing_Feature_Model.py

Requirements
------------
This script uses:
- numpy
- pandas
- matplotlib
- scikit-learn
