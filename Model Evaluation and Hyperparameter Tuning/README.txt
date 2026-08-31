Model Evaluation and Hyperparameter Tuning - Homework 5
====================================================================
This project uses the Breast Cancer Wisconsin dataset to evaluate a machine learning classifier, tune hyperparameters, analyze learning and validation curves, and compare methods for handling class imbalance.

Files
=================
-Homework5.ipynb - Jupyter notebook with the code, explanations, and plots.
-Homework5.py - Standalone Python version of the notebook.
-model_evaluation_results/ - Contains generated plots, metrics, and result explanations.

Main Tasks
=================
1- Build and evaluate a baseline model using StandardScaler, PCA, and Logistic Regression.
2- Create learning and validation curves.
3- Compare GridSearchCV and RandomizedSearchCV for hyperparameter tuning.
4- Test different methods for handling class imbalance.
5- Compare model performance using accuracy, recall, F1-score, and ROC-AUC.

Results
=================
The model achieves about 97% accuracy and ROC-AUC scores around 0.997-0.998. 
GridSearchCV and RandomizedSearchCV produce similar results, and the model shows little evidence of overfitting. 
Resampling strategies have only a small effect on performance.

Dataset
=================
569 samples, 30 features, malignant vs. benign classification, with an 80% training and 20% testing split.

Run
=================
jupyter notebook Homework5.ipynb

or

python Homework5.py

Requirements
=================
pip install scikit-learn pandas numpy matplotlib scipy

GitHub: https://github.com/Prevyrus/CS-projects
