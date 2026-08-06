Binary Classification Project
===========================

This script demonstrates binary classification using two simple neural-style classifiers:
Perceptron and Adaline. The goal is to compare how these models behave on clean and noisy
label data using the Iris dataset.

What the code does
-----------------
The program:
- Loads the Iris dataset from the UCI Machine Learning Repository if it is not already present.
- Uses only the Setosa and Versicolor classes, which makes the task binary classification.
- Uses two features: sepal length and petal length.
- Standardizes the input data before training.
- Trains a Perceptron classifier and an Adaline classifier on both clean and slightly noisy labels.
- Compares their accuracy and visualizes the decision boundaries.

Main components
--------------
- Perceptron: A simple linear classifier that updates its weights during training.
- AdalineGD: A linear classifier trained using gradient descent and mean squared error.
- plot_decision_regions(): Creates visual plots of the decision boundary for each model.

Outputs
-------
The script creates several plots and an animation in an outputs folder, including:
- Clean and noisy data visualizations
- Perceptron convergence plots
- Adaline loss plots
- Decision boundary comparison charts
- Accuracy comparison bar chart
- A GIF animation showing how the Perceptron decision boundary evolves over epochs

How to run
----------
Run the Python script:

python Binary_Classification.py

Requirements
------------
This script uses:
- numpy
- pandas
- matplotlib
- scikit-learn

Notes
-----
The dataset is downloaded automatically from the UCI repository if the local file is missing.
The script saves its generated graphics into the outputs folder so the results are easy to review.
