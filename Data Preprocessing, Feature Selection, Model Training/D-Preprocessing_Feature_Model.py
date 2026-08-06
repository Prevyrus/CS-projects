import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


"""
I did not understand the instruction very well since the assignment 
did not continue using the suggested requested dataset pulled from a database (Kaggle, UCI, Seaborn,etc).
In later steps, it goes back to the steps using the wine dataset.
"""

# Create a folder where all assignment plots will be saved
plots_folder = "homework4plots"
os.makedirs(plots_folder, exist_ok=True)

#---------------------------------------------------------------
# Load the Wine Dataset
wine = load_wine()

df_wine = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

df_wine.insert(
    0,
    "Class label",
    wine.target
)

print("Wine dataset preview:")
print(df_wine.head())

print("\nOriginal Wine shape:", df_wine.shape)


# Part 1: Handling Missing Data

# Make a copy so the original Wine dataset stays unchanged
df_missing = df_wine.copy()

# Randomly add missing values to the feature columns
np.random.seed(1)

feature_columns = df_missing.columns[1:]

# Randomly introduce missing values into the Wine features
for _ in range(20):
    row = np.random.randint(len(df_missing))
    column = np.random.choice(feature_columns)
    df_missing.loc[row, column] = np.nan

print("\nMissing values in each column:")
print(df_missing.isnull().sum())

# Compare dropping rows and columns
drop_rows = df_missing.dropna(axis=0)
drop_columns = df_missing.dropna(axis=1)

# Impute the feature columns and keep the class label unchanged
X_missing = df_missing.iloc[:, 1:]

imputer = SimpleImputer(strategy="mean")
imputed_data = imputer.fit_transform(X_missing)

imputed_df = pd.DataFrame(
    imputed_data,
    columns=X_missing.columns
)

imputed_df.insert(
    0,
    "Class label",
    df_missing["Class label"].values
)

print("\nShape with missing values:", df_missing.shape)
print("Shape after dropping rows:", drop_rows.shape)
print("Shape after dropping columns:", drop_columns.shape)
print("Shape after mean imputation:", imputed_df.shape)



#---------------------------------------------------------------
# Part 1: Categorical Data Encoding

# Create a small dataset with nominal, ordinal, and numerical data
df_categories = pd.DataFrame([
    ["green", "M", 10.1],
    ["red", "L", 13.5],
    ["blue", "XL", 15.3]
], columns=["color", "size", "price"])

# One-hot encode color because colors do not have a natural order
try:
    color_encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    color_encoder = OneHotEncoder(sparse=False)

color_encoded = color_encoder.fit_transform(
    df_categories[["color"]]
)

color_encoded_df = pd.DataFrame(
    color_encoded,
    columns=color_encoder.get_feature_names_out(["color"])
)

# Ordinal encode size because M, L, and XL have a natural order
size_encoder = OrdinalEncoder(
    categories=[["M", "L", "XL"]]
)

size_encoded = size_encoder.fit_transform(
    df_categories[["size"]]
)

size_encoded_df = pd.DataFrame(
    size_encoded,
    columns=["size_encoded"]
)

print("\nOriginal categorical data:")
print(df_categories)

print("\nOneHotEncoder output:")
print(color_encoded_df)

print("\nOrdinalEncoder output:")
print(size_encoded_df)



#---------------------------------------------------------------
# Prepare Wine Data for Modeling

X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

feature_names = df_wine.columns[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=0,
    stratify=y
)



#---------------------------------------------------------------
# Part 2: Standardization and Normalization

# Normalize the Wine features to values between 0 and 1
mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardize the Wine features around a mean of 0
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print("\nFirst normalized Wine training row:")
print(np.round(X_train_norm[0], 3))

print("\nFirst standardized Wine training row:")
print(np.round(X_train_std[0], 3))



#---------------------------------------------------------------
# Part 3: Sequential Backward Selection


# This SBS class follows the Chapter 4 example
class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self._calc_score(
            X_train,
            y_train,
            X_test,
            y_test,
            self.indices_
        )

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            # Try every subset that removes one feature
            for subset in combinations(
                self.indices_,
                r=dim - 1
            ):
                score = self._calc_score(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    subset
                )

                scores.append(score)
                subsets.append(subset)

            # Keep the subset with the best accuracy
            best = np.argmax(scores)

            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])

            dim -= 1

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        indices
    ):
        self.estimator.fit(
            X_train[:, indices],
            y_train
        )

        y_pred = self.estimator.predict(
            X_test[:, indices]
        )

        return self.scoring(
            y_test,
            y_pred
        )


# Train KNN using all standardized Wine features
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_std, y_train)

training_accuracy_all = knn.score(
    X_train_std,
    y_train
)

test_accuracy_all = knn.score(
    X_test_std,
    y_test
)

# Run SBS until only one feature remains
sbs = SBS(
    knn,
    k_features=1
)

sbs.fit(
    X_train_std,
    y_train
)

feature_counts = [
    len(subset)
    for subset in sbs.subsets_
]

plt.plot(
    feature_counts,
    sbs.scores_,
    marker="o"
)

plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.title("SBS Accuracy by Number of Wine Features")
plt.grid()

plt.savefig(
    os.path.join(
        plots_folder,
        "sbs_accuracy.png"
    )
)

plt.close()

# Find the feature subset with the highest SBS score
best_position = int(
    np.argmax(sbs.scores_)
)

best_subset = list(
    sbs.subsets_[best_position]
)

best_feature_names = list(
    feature_names[best_subset]
)

# Train KNN again using only the selected features
knn.fit(
    X_train_std[:, best_subset],
    y_train
)

training_accuracy_selected = knn.score(
    X_train_std[:, best_subset],
    y_train
)

test_accuracy_selected = knn.score(
    X_test_std[:, best_subset],
    y_test
)

print("\nBest number of SBS features:", len(best_subset))
print("Best SBS features:", best_feature_names)

print("\nKNN accuracy with all Wine features")
print("Training accuracy:", round(training_accuracy_all, 3))
print("Test accuracy:", round(test_accuracy_all, 3))

print("\nKNN accuracy after SBS")
print("Training accuracy:", round(training_accuracy_selected, 3))
print("Test accuracy:", round(test_accuracy_selected, 3))


#---------------------------------------------------------------
# Part 4: Random Forest Feature Importance

# Train Random Forest on the Wine dataset
forest = RandomForestClassifier(
    n_estimators=500,
    random_state=1
)

forest.fit(
    X_train,
    y_train
)

importances = forest.feature_importances_

indices = np.argsort(
    importances
)[::-1]

print("\nRandom Forest feature ranking:")

for feature_number in range(
    X_train.shape[1]
):
    print(
        "%2d) %-30s %f" % (
            feature_number + 1,
            feature_names[indices[feature_number]],
            importances[indices[feature_number]]
        )
    )

plt.bar(
    range(X_train.shape[1]),
    importances[indices],
    align="center"
)

plt.xticks(
    range(X_train.shape[1]),
    feature_names[indices],
    rotation=90
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Wine Random Forest Feature Importance")

plt.savefig(
    os.path.join(
        plots_folder,
        "random_forest_feature_importance.png"
    )
)

plt.close()

top_three_features = list(
    feature_names[indices[:3]]
)

print("\nTop 3 Random Forest features:", top_three_features)
print("Best SBS features:", best_feature_names)



#---------------------------------------------------------------
# Bonus: L1 Regularization

weights = []
parameters = []

# Try different C values and save the feature coefficients
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(
        l1_ratio=1.0,
        C=10.0 ** c,
        solver="saga",
        random_state=0,
        max_iter=5000
    )

    lr.fit(
        X_train_std,
        y_train
    )

    weights.append(
        lr.coef_[1]
    )

    parameters.append(
        10.0 ** c
    )

weights = np.array(weights)

# Plot how each Wine feature coefficient changes
for column in range(
    weights.shape[1]
):
    plt.plot(
        parameters,
        weights[:, column],
        label=feature_names[column]
    )

plt.axhline(0, linestyle="--")

plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Weight coefficient")
plt.title("Wine L1 Logistic Regression Coefficients")
plt.xscale("log")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.savefig(
    os.path.join(
        plots_folder,
        "l1_regularization.png"
    ),
    bbox_inches="tight"
)

plt.close()


#---------------------------------------------------------------
#Print message that save the plots on the right folder

print("\nAll plots were saved in the homework4plots folder.")
