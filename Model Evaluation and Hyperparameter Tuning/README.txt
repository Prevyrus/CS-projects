Model Evaluation and Hyperparameter Tuning - Homework 5

What's This Project About?
--------------------------
This is a machine learning project focused on evaluating model performance and finding 
the best hyperparameters for a classifier. Using the Breast Cancer Wisconsin dataset,
the project explores different techniques to improve and assess model accuracy.

Files in This Folder
--------------------

1. Homework5.ipynb
   - Jupyter notebook with step-by-step code and explanations
   - Best for understanding the concepts - has markdown cells explaining each part
   - Can run interactively and see plots right in the notebook
   - Good for learning and reference

2. Homework5.py
   - Same code as the notebook but as a standalone Python script
   - Just run it: python Homework5.py
   - All output and plots get saved to the model_evaluation_results folder
   - No extra explanations, just the code

3. model_evaluation_results/
   - Output folder containing all generated results
   
   Files inside:
   - learning_curve.png: Shows how accuracy changes as you add more training data
   - validation_curve_pca.png: Shows the effect of reducing dimensions (PCA)
   - roc_curves_resampling.png: Compares 3 different data balancing strategies
   - resampling_comparison.csv: Table with performance metrics for each strategy
   - explanation of perfomance.txt: Detailed analysis of the results

What Does the Code Do?
----------------------

Task 1: Create a baseline model using a pipeline (standardize data → reduce dimensions → classify)
        and check how well it performs with cross-validation

Task 2A: Learning Curve - see if model improves with more training data
         (check for underfitting or overfitting)

Task 2B: Validation Curve - find the best number of dimensions to use (PCA components)

Task 3A: GridSearchCV - try every combination of hyperparameters to find the best one
         (takes longer but guaranteed to find the best)

Task 3B: RandomizedSearchCV - randomly try different hyperparameters
         (faster but might not find the absolute best)

Task 4: Class Imbalance - deal with the fact that there are more benign cases than malignant
        Try different strategies: original data, remove some samples, add copies of minority class

Task 5: ROC Curves - compare how well each strategy works for predicting cancer

How to Run It
-------------

With Jupyter Notebook (recommended for learning):
  jupyter notebook Homework5.ipynb

As a Python script:
  python Homework5.py

Requirements:
  pip install scikit-learn pandas numpy matplotlib scipy

What You'll Find in Results
---------------------------

- learning_curve.png: The model works well, training accuracy is high and 
  validation accuracy catches up as more data is added (good sign)

- validation_curve_pca.png: Around 10-15 dimensions is the sweet spot.
  Too few = bad performance, too many = no improvement

- resampling_comparison.csv: Shows accuracy, recall, and F1-scores for 
  different data balancing methods

- roc_curves_resampling.png: All three strategies perform really well 
  (AUC around 0.997-0.998, which is excellent)

Key Takeaways
-------------
- Model achieves ~97% accuracy
- Both GridSearchCV and RandomizedSearchCV find similarly good hyperparameters
- The model generalizes well and doesn't overfit much
- Class imbalance doesn't hurt performance much on this dataset
- More data helps the model improve (learning curve shows this)

Dataset Info
------------
- 569 breast cancer samples
- 30 features (measurements from cell images)
- Predict: malignant (cancer) vs benign (not cancer)
- Split: 80% training, 20% testing

Questions?
----------
Check Homework5.ipynb for detailed explanations of each concept and step.
GitHub: https://github.com/Prevyrus/CS-projects
