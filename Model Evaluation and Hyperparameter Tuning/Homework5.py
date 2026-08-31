import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, learning_curve, train_test_split, validation_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# We will create one folder for all assignment results
output_folder = "model_evaluation_results"
os.makedirs(output_folder, exist_ok=True)

random_state = 1


# 1. Load and prepare the Breast Cancer Wisconsin dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("Breast Cancer Wisconsin dataset")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Class names: {list(cancer.target_names)}")
print(f"Full class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=random_state,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")


# Use the same stratified folds throughout the assignment
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=random_state,
)

### TO-DO tasks:

# 2. Task 1: Pipeline and cross-validation
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(max_iter=10000, random_state=random_state),
)

cv_scores = cross_val_score(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
)

pipe_lr.fit(X_train, y_train)
baseline_test_accuracy = pipe_lr.score(X_test, y_test)

print("\nTask 1: Pipeline and cross-validation")
print(f"Cross-validation scores: {np.round(cv_scores, 3)}")
print( f"Mean CV accuracy: {cv_scores.mean():.3f} " f"+/- {cv_scores.std():.3f}")
print(f"Baseline test accuracy: {baseline_test_accuracy:.3f}")


# 3. Task 2A: Learning curve
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)
validation_std = np.std(validation_scores, axis=1)

plt.plot(train_sizes, train_mean, marker="o", label="Training accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)

plt.plot(train_sizes, validation_mean, marker="s", linestyle="--", label="Validation accuracy")
plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.15)

plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve: PCA and Logistic Regression")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "learning_curve.png"))
plt.close()


# 4. Task 2B: Validation curve for the number of PCA components
pca_components = [2, 5, 10, 15, 20, 25, 30]

validation_train_scores, validation_test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name="pca__n_components",
    param_range=pca_components,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
)

validation_train_mean = np.mean(validation_train_scores, axis=1)
validation_train_std = np.std(validation_train_scores, axis=1)
validation_test_mean = np.mean(validation_test_scores, axis=1)
validation_test_std = np.std(validation_test_scores, axis=1)

plt.plot(pca_components, validation_train_mean, marker="o", label="Training accuracy")
plt.fill_between(pca_components, validation_train_mean - validation_train_std, validation_train_mean + validation_train_std, alpha=0.15)

plt.plot(pca_components, validation_test_mean, marker="s", linestyle="--", label="Validation accuracy")
plt.fill_between(pca_components, validation_test_mean - validation_test_std, validation_test_mean + validation_test_std, alpha=0.15)

plt.xlabel("Number of PCA components")
plt.ylabel("Accuracy")
plt.title("Validation Curve: Number of PCA Components")
plt.xticks(pca_components)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, "validation_curve_pca.png"))
plt.close()

best_pca_index = int(np.argmax(validation_test_mean))
best_pca_components = pca_components[best_pca_index]

print("\nTask 2: Learning and validation curves")
print("Saved learning_curve.png")
print("Saved validation_curve_pca.png")
print(f"Best PCA component count from the validation curve: "f"{best_pca_components}")
print(f"Best mean validation accuracy: " f"{validation_test_mean[best_pca_index]:.3f}")


# 5. Task 3A: GridSearchCV for Logistic Regression
search_pipeline = make_pipeline(
    StandardScaler(),
    PCA(),
    LogisticRegression(max_iter=10000, random_state=random_state),
)

grid_parameters = {
    "pca__n_components": [5, 10, 15, 20, 25, 30],
    "logisticregression__solver": ["liblinear", "lbfgs"],
    "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}

grid_search = GridSearchCV(
    estimator=search_pipeline,
    param_grid=grid_parameters,
    scoring="accuracy",
    cv=cv,
    refit=True,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
grid_test_accuracy = grid_search.score(X_test, y_test)

print("\nTask 3A: GridSearchCV")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Grid search test accuracy: {grid_test_accuracy:.3f}")


# 6. Task 3B: RandomizedSearchCV for comparison
random_parameters = {
    "pca__n_components": [5, 10, 15, 20, 25, 30],
    "logisticregression__solver": ["liblinear", "lbfgs"],
    "logisticregression__C": loguniform(0.001, 100.0),
}

random_search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=random_parameters,
    n_iter=30,
    scoring="accuracy",
    cv=cv,
    refit=True,
    random_state=random_state,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)
random_test_accuracy = random_search.score(X_test, y_test)

print("\nTask 3B: RandomizedSearchCV")
print(f"Best CV accuracy: {random_search.best_score_:.3f}")
print(f"Best parameters: {random_search.best_params_}")
print(f"Randomized search test accuracy: {random_test_accuracy:.3f}")


# 7. Task 4: Compare original, downsampled, and upsampled training data
# Only the training set is resampled. The test set stays unchanged.
train_data = np.column_stack((X_train, y_train))
class_0 = train_data[train_data[:, -1] == 0]
class_1 = train_data[train_data[:, -1] == 1]

if len(class_0) < len(class_1):
    minority_class = class_0
    majority_class = class_1
else:
    minority_class = class_1
    majority_class = class_0

# Downsample the majority class to the minority class size.
majority_downsampled = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=random_state,
)

downsampled_data = np.vstack((minority_class, majority_downsampled))
rng = np.random.default_rng(random_state)
rng.shuffle(downsampled_data)

X_train_downsampled = downsampled_data[:, :-1]
y_train_downsampled = downsampled_data[:, -1].astype(int)

# Upsample the minority class to the majority class size.
minority_upsampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=random_state,
)

upsampled_data = np.vstack((majority_class, minority_upsampled))
rng.shuffle(upsampled_data)

X_train_upsampled = upsampled_data[:, :-1]
y_train_upsampled = upsampled_data[:, -1].astype(int)


def evaluate_resampled_model(name, X_training, y_training):
    model = grid_search.best_estimator_
    model.fit(X_training, y_training)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=cancer.target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "Training data": name,
        "Training samples": len(y_training),
        "Class 0 samples": int(np.sum(y_training == 0)),
        "Class 1 samples": int(np.sum(y_training == 1)),
        "Accuracy": accuracy,
        "Malignant recall": report["malignant"]["recall"],
        "Benign recall": report["benign"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Confusion matrix": confusion_matrix(y_test, predictions).tolist(),
    }


resampling_results = [
    evaluate_resampled_model("Original", X_train, y_train),
    evaluate_resampled_model(
        "Majority downsampled",
        X_train_downsampled,
        y_train_downsampled,
    ),
    evaluate_resampled_model(
        "Minority upsampled",
        X_train_upsampled,
        y_train_upsampled,
    ),
]

results_table = pd.DataFrame(resampling_results)
results_table.to_csv(
    os.path.join(output_folder, "resampling_comparison.csv"),
    index=False,
)

print("\nTask 4: Class imbalance and resampling")
print(
    results_table[
        [
            "Training data",
            "Training samples",
            "Class 0 samples",
            "Class 1 samples",
            "Accuracy",
            "Malignant recall",
            "Benign recall",
            "Macro F1",
        ]
    ].round(3).to_string(index=False)
)

for result in resampling_results:
    print(f"{result['Training data']} confusion matrix: " f"{result['Confusion matrix']}")

# 8. ROC curves for original and resampled models

roc_datasets = [
    ("Original", X_train, y_train),
    ("Majority downsampled", X_train_downsampled, y_train_downsampled),
    ("Minority upsampled", X_train_upsampled, y_train_upsampled),
]

for name, X_training, y_training in roc_datasets:
    model = grid_search.best_estimator_

    model.fit(X_training, y_training)

    # Class 0 is malignant
    probabilities = model.predict_proba(X_test)[:, 0]

    fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=0)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")


plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing (AUC = 0.500)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Resampling Comparison")

plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_folder, "roc_curves_resampling.png"))
plt.close()

print("Saved roc_curves_resampling.png")

# 9. Save a short text summary
best_resampling_result = max(
    resampling_results,
    key=lambda result: result["Macro F1"])

print
(
    f"\nBest resampling strategy based on Macro F1: "
    f"{best_resampling_result['Training data']} "
    f"(Macro F1: {best_resampling_result['Macro F1']:.3f})"
)
print(f"\nAll results were saved in: {output_folder}")