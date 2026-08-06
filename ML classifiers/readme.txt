ML Classifiers Project
=====================

This project compares several common machine learning classifiers on the Iris dataset.
The script trains and evaluates multiple models to show how different classifiers perform on
binary/ multiclass classification tasks.

What the code does
------------------
The script:
- Loads the Iris dataset from scikit-learn.
- Uses petal length and petal width as the input features.
- Splits the data into training and testing sets.
- Standardizes the feature values where needed.
- Trains and evaluates the following classifiers:
  - Perceptron
  - Logistic Regression
  - SVM with a linear kernel
  - SVM with an RBF kernel
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
- Creates decision boundary plots and confusion matrices for each model.
- Saves the final comparison results to a CSV file.

Outputs
-------
The script generates:
- Plot images in the homework3plots folder
- A CSV file named iris_classifier_comparison.csv
- Printed accuracy and confusion matrix results in the console

How to run
----------
Run the script with:

python ML_classifiers.py

Requirements
------------
This script uses:
- numpy
- pandas
- matplotlib
- scikit-learn
