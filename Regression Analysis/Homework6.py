import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, RANSACRegressor, Ridge)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# We need to save all the plots in the folder to include them later
output_folder = "regression_analysis_results"
os.makedirs(output_folder, exist_ok=True)

random_state = 1


# ---------------------------------------------------------------------------------
# Task 1: Data Preprocessing
# Steps:
# 1. load the Ames Housing dataset and handle the missing values
# 2. encode Central Air as 0 for no and 1 for yes
# 3. split the data into 80% training data and 20% testing data
# 4. standardize the feature values using StandardScaler

# We use the same columns that were used in the Chapter 9 code
columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

#  load the Ames Housing dataset from the same source used in the textbook.
df = pd.read_csv(
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt",
    sep="\t",
    usecols=columns,
)

print("Task 1: Data Preprocessing")
print("Missing values before cleaning:")
print(df.isnull().sum())

# Central Air is the categorical variable
df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})

# remove the rows that still have missing values so every model receives complete numerical observations
df = df.dropna().copy()

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(f"\nRows available after cleaning: {len(df)}")
print("\nFirst five rows:")
print(df.head())

feature_names = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
]

target_name = "SalePrice"

X = df[feature_names].values
y = df[target_name].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=random_state,
)

# we have to fit the scaler only on the training data so information from the test set does not influence the models
feature_scaler = StandardScaler()
X_train_std = feature_scaler.fit_transform(X_train)
X_test_std = feature_scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")


# ----------------------------------------------------------------------------------
# Task 2: Implement Linear Regression with Gradient Descent
# Steps:
# 1. complete the LinearRegressionGD class using the textbook structure
# 2. use Gr Liv Area to predict SalePrice
# 3. plot the training loss across the epochs
# 4. compare the actual and predicted sale prices on the test data


class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random_generator = np.random.RandomState(self.random_state)
        self.w_ = random_generator.normal(
            loc=0.0,
            scale=0.01,
            size=X.shape[1],
        )
        self.b_ = np.array([0.0])
        self.losses_ = []

        # During each epoch, we need to update the weight and intercept to lower the mean squared error
        for _ in range(self.n_iter):
            predictions = self.net_input(X)
            errors = y - predictions

            self.w_ += (
                self.eta
                * 2.0
                * X.T.dot(errors)
                / X.shape[0]
            )
            self.b_ += self.eta * 2.0 * errors.mean()

            loss = (errors ** 2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


X_simple = df[["Gr Liv Area"]].values
y_simple = df["SalePrice"].values

X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple,
    y_simple,
    test_size=0.20,
    random_state=random_state,
)

# Gradient descent works better when the input and target are on a similar scale, so we standardize both before training
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_simple_train_std = x_scaler.fit_transform(X_simple_train)
X_simple_test_std = x_scaler.transform(X_simple_test)
y_simple_train_std = y_scaler.fit_transform(
    y_simple_train.reshape(-1, 1)
).flatten()

lr_gd = LinearRegressionGD(
    eta=0.1,
    n_iter=50,
    random_state=random_state,
)
lr_gd.fit(X_simple_train_std, y_simple_train_std)

plt.plot(range(1, lr_gd.n_iter + 1), lr_gd.losses_)
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("Gradient Descent Training Loss")
plt.savefig(os.path.join(output_folder, "gradient_descent_loss.png"))
plt.close()

# we convert the standardized predictions back to sale prices in dollars so the results are easier to interpret
y_gd_test_std = lr_gd.predict(X_simple_test_std)
y_gd_test_pred = y_scaler.inverse_transform(
    y_gd_test_std.reshape(-1, 1)
).flatten()

minimum_value = min(y_simple_test.min(), y_gd_test_pred.min())
maximum_value = max(y_simple_test.max(), y_gd_test_pred.max())

plt.scatter(y_simple_test, y_gd_test_pred)
plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], label="Perfect prediction")
plt.xlabel("Actual sale price")
plt.ylabel("Predicted sale price")
plt.title("Gradient Descent: Actual vs. Predicted")
plt.legend()
plt.savefig(os.path.join(output_folder, "gradient_descent_predictions.png"))
plt.close()

print("\nTask 2: Linear Regression with Gradient Descent")
print(
    f"Gradient Descent MSE: "
    f"{mean_squared_error(y_simple_test, y_gd_test_pred):,.2f}"
)
print(
    f"Gradient Descent MAE: "
    f"{mean_absolute_error(y_simple_test, y_gd_test_pred):,.2f}"
)
print(
    f"Gradient Descent R-squared: "
    f"{r2_score(y_simple_test, y_gd_test_pred):.3f}"
)


# -------------------------------------------------------------------------
# Task 3: Implement and Compare Scikit-Learn Models
# Steps:
# 1. train OLS, RANSAC, Decision Tree, and Random Forest models
# 2. evaluate every model on the same test set
# 3. compare them using MSE, MAE, and R-squared


def evaluate_regression_model(
    name,
    model,
    X_training,
    X_testing,
):
    #Train one model and return its test-set regression metrics
    model.fit(X_training, y_train)
    predictions = model.predict(X_testing)

    return {
        "Model": name,
        "MSE": mean_squared_error(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "R-squared": r2_score(y_test, predictions),
    }


ols_model = LinearRegression()
ransac_model = RANSACRegressor(
    estimator=LinearRegression(),
    random_state=random_state,
)
decision_tree_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=random_state,
)
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=random_state,
    n_jobs=-1,
)

# we use standardized features for the linear models. Tree models do not require scaling, so we use the original values
model_results = [
    evaluate_regression_model(
        "Ordinary Least Squares",
        ols_model,
        X_train_std,
        X_test_std,
    ),
    evaluate_regression_model(
        "RANSAC Regression",
        ransac_model,
        X_train_std,
        X_test_std,
    ),
    evaluate_regression_model(
        "Decision Tree Regression",
        decision_tree_model,
        X_train,
        X_test,
    ),
    evaluate_regression_model(
        "Random Forest Regression",
        random_forest_model,
        X_train,
        X_test,
    ),
]

model_results_table = pd.DataFrame(model_results)

print("\nTask 3: Scikit-Learn Regression Models")
print(model_results_table.round({"MSE": 2, "MAE": 2, "R-squared": 3}).to_string(index=False))


# --------------------------------------------------------
# Task 4: Explore Regularization Techniques
# Steps:
# 1. train Lasso, Ridge, and ElasticNet regression models
# 2. compare their MSE, MAE, and R-squared values
# 3. plot the coefficients to see how regularization changes the features

regularization_models = {
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=random_state,
    ),
}

regularization_results = []
coefficient_results = {}

for model_name, model in regularization_models.items():
    model.fit(X_train_std, y_train)
    predictions = model.predict(X_test_std)

    regularization_results.append(
        {
            "Model": model_name,
            "MSE": mean_squared_error(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "R-squared": r2_score(y_test, predictions),
        }
    )
    coefficient_results[model_name] = model.coef_

regularization_results_table = pd.DataFrame(regularization_results)

print("\nTask 4: Regularization Techniques")
print(regularization_results_table.round({"MSE": 2, "MAE": 2, "R-squared": 3}).to_string(index=False))

# we place all three coefficient lines on one plot so we can compare how each method shrinks the coefficients
for model_name, coefficients in coefficient_results.items():
    plt.plot(feature_names, coefficients, label=model_name)

plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Regularized Regression Coefficients")
plt.legend()
plt.savefig(os.path.join(output_folder, "regularization_coefficients.png"))
plt.close()

print(
    "Lasso can shrink some coefficients all the way to zero. Ridge usually "
    "shrinks every coefficient, while ElasticNet combines both approaches."
)


# ------------------------------------------------------------------
# Task 5: Polynomial Regression
# Steps:
# 1. create linear, quadratic, and cubic versions of Gr Liv Area
# 2. fit each model and plot its predicted sale-price curve
# 3. compare the models using test-set MSE and R-squared

X_poly = df[["Gr Liv Area"]].values
y_poly = df["SalePrice"].values

X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
    X_poly,
    y_poly,
    test_size=0.20,
    random_state=random_state,
)

polynomial_results = []
polynomial_models = {}

for degree in [1, 2, 3]:
    polynomial_transformer = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )

    X_train_degree = polynomial_transformer.fit_transform(X_poly_train)
    X_test_degree = polynomial_transformer.transform(X_poly_test)

    polynomial_model = LinearRegression()
    polynomial_model.fit(X_train_degree, y_poly_train)
    test_predictions = polynomial_model.predict(X_test_degree)

    polynomial_results.append(
        {
            "Degree": degree,
            "Model": ["Linear", "Quadratic", "Cubic"][degree - 1],
            "MSE": mean_squared_error(y_poly_test, test_predictions),
            "R-squared": r2_score(y_poly_test, test_predictions),
        }
    )

    polynomial_models[degree] = (
        polynomial_transformer,
        polynomial_model,
    )

polynomial_results_table = pd.DataFrame(polynomial_results)

print("\nTask 5: Polynomial Regression")
print(
    polynomial_results_table.round(
        {"MSE": 2, "R-squared": 3}
    ).to_string(index=False)
)

X_plot = np.linspace(
    X_poly.min(),
    X_poly.max(),
    500,
).reshape(-1, 1)

plt.scatter(X_poly_train, y_poly_train, label="Training data")

for degree, (transformer, model) in polynomial_models.items():
    predicted_curve = model.predict(transformer.transform(X_plot))
    model_name = ["Linear", "Quadratic", "Cubic"][degree - 1]
    plt.plot(X_plot, predicted_curve, label=model_name)

plt.xlabel("Living area above ground in square feet")
plt.ylabel("Sale price")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.savefig(os.path.join(output_folder, "polynomial_regression.png"))
plt.close()


# --------------------------------------------------
# Task 6: Conclusion
# Steps:
# 1. use the test results to identify the strongest model from each comparison 
# 2. summarize what happened when we added regularization and polynomial terms

all_regression_results = pd.concat(
    [model_results_table, regularization_results_table],
    ignore_index=True,
)

best_overall = all_regression_results.loc[
    all_regression_results["R-squared"].idxmax()
]
best_regularized = regularization_results_table.loc[
    regularization_results_table["R-squared"].idxmax()
]
best_polynomial = polynomial_results_table.loc[
    polynomial_results_table["R-squared"].idxmax()
]

print("\nTask 6: Conclusion")
print(
    f"\nThe best overall Scikit-Learn model was {best_overall['Model']} "
    f"with a test R-squared of {best_overall['R-squared']:.3f}."
)
print(
    f"Among the regularized models, {best_regularized['Model']} performed "
    f"best with a test R-squared of {best_regularized['R-squared']:.3f}."
)
print(
    "The regularization methods reduced the size of the coefficients. "
    "Lasso can remove a feature by setting its coefficient to zero, Ridge "
    "keeps all features but shrinks them, and ElasticNet combines both ideas."
)
print(
    f"For polynomial regression, the {best_polynomial['Model']} model had "
    f"the best test R-squared ({best_polynomial['R-squared']:.3f})."
)
print(
    "The polynomial terms let the model follow curved relationships that a "
    "straight line cannot capture. However, adding more terms does not always "
    "improve test performance and can lead to overfitting."
)

#Save the results from the print statements to a text file
with open(os.path.join(output_folder, "conclusion.txt"), "w") as f:
    f.write("\nTask 6: Conclusion\n")
    f.write(
        f"\nThe best overall Scikit-Learn model was {best_overall['Model']} "
        f"with a test R-squared of {best_overall['R-squared']:.3f}.\n"
    )
    f.write(
        f"Among the regularized models, {best_regularized['Model']} performed "
        f"best with a test R-squared of {best_regularized['R-squared']:.3f}.\n"
    )
    f.write(
        "The regularization methods reduced the size of the coefficients. "
        "Lasso can remove a feature by setting its coefficient to zero, Ridge "
        "keeps all features but shrinks them, and ElasticNet combines both ideas.\n"
    )
    f.write(
        f"For polynomial regression, the {best_polynomial['Model']} model had "
        f"the best test R-squared ({best_polynomial['R-squared']:.3f}).\n"
    )
    f.write(
        "The polynomial terms let the model follow curved relationships that a "
        "straight line cannot capture. However, adding more terms does not always "
        "improve test performance and can lead to overfitting.\n"
    )


# we save the model tables so we can include the exact results in our submission.
all_regression_results.to_csv(
    os.path.join(output_folder, "regression_model_comparison.csv"),
    index=False,
)
polynomial_results_table.to_csv(
    os.path.join(output_folder, "polynomial_model_comparison.csv"),
    index=False,
)

print(f"\nThe tables and plots were saved in: {output_folder}")