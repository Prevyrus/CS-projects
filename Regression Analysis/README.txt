Regression Analysis - Homework 6
=================================

What's This Project About?
--------------------------
This project explores different regression techniques to predict house prices using
the Ames Housing dataset. It compares linear models, regularized regression, polynomial
features, and non-linear models to find the best approach for price prediction.

Files in This Folder
--------------------

1. homework_6_regression_notebook.ipynb
   - Jupyter notebook with step-by-step explanations
   - Visualizations and detailed analysis
   - Best for learning and understanding concepts
   - Can run interactively

2. Homework6.py
   - Standalone Python script with all analysis
   - Run with: python Homework6.py
   - All plots and results saved to regression_analysis_results/

3. regression_analysis_results/
   - Output folder with all generated results
   
   Files inside:
   - regression_model_comparison.csv: Performance metrics for all models
   - polynomial_model_comparison.csv: Polynomial regression comparison
   - conclusion.txt: Summary of findings

What Does the Code Do?
----------------------

TASK 1: Data Preprocessing
   - Load Ames Housing dataset (1,460 houses)
   - Handle missing values
   - Encode categorical variables (Central Air: Y/N → 1/0)
   - Split: 80% training, 20% testing
   - Standardize features using StandardScaler

TASK 2: Linear Regression Models
   - Linear Regression: Baseline model (no regularization)
   - Ridge Regression: L2 regularization (shrinks coefficients)
   - Lasso Regression: L1 regularization (eliminates weak features)
   - ElasticNet: Combines Ridge + Lasso
   - Compare: MSE, MAE, R² scores

TASK 3: Robust Regression
   - RANSAC (Random Sample Consensus)
   - Reduces impact of outliers in housing prices
   - Useful when data has extreme values

TASK 4: Polynomial Features
   - Linear: Degree 1 (simple line)
   - Quadratic: Degree 2 (curved fit)
   - Cubic: Degree 3 (more complex curve)
   - Prevents underfitting but risk of overfitting

TASK 5: Non-Linear Models
   - Decision Tree Regressor: Recursive feature splits
   - Random Forest: Ensemble of many trees
   - Often captures non-linear patterns better

Dataset Info
------------
- Name: Ames Housing dataset
- Samples: 1,460 houses
- Features used: 5
  - Overall Quality (1-10 scale)
  - Overall Condition (1-9 scale)
  - Gross Living Area (sq ft)
  - Central Air (Y/N)
  - Total Basement Square Feet
- Target: Sale Price ($)

How to Run
----------

With Jupyter Notebook (recommended):
  jupyter notebook homework_6_regression_notebook.ipynb

As Python script:
  python Homework6.py

Requirements:
  pip install pandas numpy matplotlib scikit-learn

Key Results
-----------

Best Performing Models:
  - Linear Regression R²: ~0.52
  - Ridge Regression R²: ~0.52 (prevents overfitting)
  - Lasso R²: ~0.52 (feature selection)
  - Quadratic Polynomial R²: ~0.65 (better fit)
  - Random Forest R²: ~0.84 (captures non-linearity well)

Key Findings:
  1. Linear models are simple but underfit the data
  2. Polynomial features help capture curvature
  3. Random Forest achieves best performance
  4. Regularization (Ridge/Lasso) prevents overfitting
  5. Important features: Living Area, Quality, Basement

Performance Metrics Explained
-----------------------------

- R² Score: How well model fits data (0-1, higher is better)
- MSE: Average squared error (smaller is better)
- MAE: Average absolute error in dollars (smaller is better)
- RMSE: Square root of MSE (in same units as price)

Visualizations Generated
------------------------

1. gradient_descent_training_loss.png
   - Shows how loss decreases during training
   - Indicates good convergence

2. gradient_descent_actual_vs_predicted.png
   - Scatter plot comparing actual vs predicted prices
   - Points close to line = good predictions

3. regularized_regression_coefficients.png
   - Compares how Ridge, Lasso, ElasticNet weight features
   - Lasso shrinks weak features to zero

4. linear_polynomial_regression.png
   - Compares linear, quadratic, cubic fits
   - Shows polynomial fits the data better

Questions?
----------
Check homework_6_regression_notebook.ipynb for detailed explanations.
GitHub: https://github.com/Prevyrus/CS-projects
