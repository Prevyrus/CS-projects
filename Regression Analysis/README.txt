Regression Analysis - Homework 6

This project uses the Ames Housing dataset to compare different regression techniques for predicting house prices, including linear, regularized, polynomial, robust, and tree-based models.

Files

homework_6_regression_notebook.ipynb - Jupyter notebook with code, explanations, and visualizations.
Homework6.py - Standalone Python version of the analysis.
regression_analysis_results/ - Contains generated plots, performance metrics, model comparisons, and conclusions.

Main Tasks

Preprocess the Ames Housing dataset and prepare features for modeling.
Compare Linear Regression, Ridge, Lasso, and ElasticNet.
Use RANSAC regression to reduce the influence of outliers.
Compare linear, quadratic, and cubic polynomial regression.
Evaluate Decision Tree and Random Forest regression models.
Compare performance using R², MSE, MAE, and RMSE.

Results

Linear and regularized regression models achieve an R² of about 0.52, while quadratic polynomial regression improves performance to about 0.65. Random Forest performs best with an R² of approximately 0.84, showing that non-linear models capture housing price patterns more effectively.

Dataset

1,460 houses with 5 selected features: Overall Quality, Overall Condition, Gross Living Area, Central Air, and Total Basement Square Feet. The target variable is Sale Price.

Run

jupyter notebook homework_6_regression_notebook.ipynb

or

python Homework6.py

Requirements

pip install pandas numpy matplotlib scikit-learn

GitHub: https://github.com/Prevyrus/CS-projects
