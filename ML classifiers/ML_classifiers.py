import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Create a folder where all assignment plots will be saved
plots_folder = "homework3plots"
os.makedirs(plots_folder, exist_ok=True)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Create a grid using the two selected features
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    # Predict the class for each point in the grid
    labels = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    labels = labels.reshape(xx1.shape)

    plt.contourf(xx1, xx2, labels, alpha=0.3)

    class_names = ["Setosa", "Versicolor", "Virginica"]

    # Plot the three Iris classes and let Matplotlib choose the colors
    for class_label in np.unique(y):
        plt.scatter(
            X[y == class_label, 0],
            X[y == class_label, 1],
            label=class_names[class_label]
        )

    # Add an outline around the test examples
    if test_idx is not None:
        X_test_plot = X[test_idx, :]
        plt.scatter(
            X_test_plot[:, 0],
            X_test_plot[:, 1],
            facecolors="none",
            edgecolors="black",
            s=100,
            label="Test set"
        )


def evaluate_model(name, y_test, y_pred):
    misclassified = (y_test != y_pred).sum()
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + name)
    print("Misclassified examples:", misclassified)
    print("Accuracy:", round(accuracy, 3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return misclassified, accuracy


def show_confusion_matrix(name, y_test, y_pred, filename):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Setosa", "Versicolor", "Virginica"]
    )
    plt.title(name + " Confusion Matrix")
    plt.savefig(os.path.join(plots_folder, filename))
    plt.close()


# Load the Iris dataset
iris = datasets.load_iris()

# Use petal length and petal width so the decision boundaries can be plotted
X = iris.data[:, [2, 3]]
y = iris.target

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1, stratify=y
)

# Standardize the models that are affected by feature scales
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Combine the data so the training and testing examples appear on the plots
X_combined = np.vstack((X_train, X_test))
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
test_indices = range(len(y_train), len(y_combined))

results = []


# 1. Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

misclassified, accuracy = evaluate_model("Perceptron", y_test, y_pred)
results.append(["Perceptron", misclassified, accuracy])

plot_decision_regions(X_combined_std, y_combined, ppn, test_indices)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.title("Perceptron")
plt.legend()
plt.savefig(os.path.join(plots_folder, "perceptron_decision_boundary.png"))
plt.close()

show_confusion_matrix("Perceptron", y_test, y_pred, "perceptron_confusion_matrix.png")


# 2. Logistic Regression
lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000, random_state=1)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

misclassified, accuracy = evaluate_model("Logistic Regression", y_test, y_pred)
results.append(["Logistic Regression", misclassified, accuracy])

plot_decision_regions(X_combined_std, y_combined, lr, test_indices)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.title("Logistic Regression")
plt.legend()
plt.savefig(os.path.join(plots_folder, "logistic_regression_decision_boundary.png"))
plt.close()

show_confusion_matrix("Logistic Regression", y_test, y_pred, "logistic_regression_confusion_matrix.png")


# 3. SVM with a linear kernel
svm_linear = SVC(kernel="linear", C=1.0, random_state=1)
svm_linear.fit(X_train_std, y_train)
y_pred = svm_linear.predict(X_test_std)

misclassified, accuracy = evaluate_model("SVM - Linear Kernel", y_test, y_pred)
results.append(["SVM - Linear Kernel", misclassified, accuracy])

plot_decision_regions(X_combined_std, y_combined, svm_linear, test_indices)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.title("SVM - Linear Kernel")
plt.legend()
plt.savefig(os.path.join(plots_folder, "svm_linear_decision_boundary.png"))
plt.close()

show_confusion_matrix("SVM - Linear Kernel", y_test, y_pred, "svm_linear_confusion_matrix.png")


# 4. SVM with an RBF kernel
svm_rbf = SVC(kernel="rbf", gamma=0.2, C=1.0, random_state=1)
svm_rbf.fit(X_train_std, y_train)
y_pred = svm_rbf.predict(X_test_std)

misclassified, accuracy = evaluate_model("SVM - RBF Kernel", y_test, y_pred)
results.append(["SVM - RBF Kernel", misclassified, accuracy])

plot_decision_regions(X_combined_std, y_combined, svm_rbf, test_indices)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.title("SVM - RBF Kernel")
plt.legend()
plt.savefig(os.path.join(plots_folder, "svm_rbf_decision_boundary.png"))
plt.close()

show_confusion_matrix("SVM - RBF Kernel", y_test, y_pred, "svm_rbf_confusion_matrix.png")


# 5. Decision Tree
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

misclassified, accuracy = evaluate_model("Decision Tree", y_test, y_pred)
results.append(["Decision Tree", misclassified, accuracy])

plot_decision_regions(X_combined, y_combined, tree_model, test_indices)
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.title("Decision Tree")
plt.legend()
plt.savefig(os.path.join(plots_folder, "decision_tree_decision_boundary.png"))
plt.close()

show_confusion_matrix("Decision Tree", y_test, y_pred, "decision_tree_confusion_matrix.png")

plot_tree(
    tree_model,
    feature_names=["Petal length", "Petal width"],
    class_names=["Setosa", "Versicolor", "Virginica"],
    filled=True
)
plt.title("Decision Tree Structure")
plt.savefig(os.path.join(plots_folder, "decision_tree_structure.png"))
plt.close()


# 6. Random Forest
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

misclassified, accuracy = evaluate_model("Random Forest", y_test, y_pred)
results.append(["Random Forest", misclassified, accuracy])

plot_decision_regions(X_combined, y_combined, forest, test_indices)
plt.xlabel("Petal length [cm]")
plt.ylabel("Petal width [cm]")
plt.title("Random Forest")
plt.legend()
plt.savefig(os.path.join(plots_folder, "random_forest_decision_boundary.png"))
plt.close()

show_confusion_matrix("Random Forest", y_test, y_pred, "random_forest_confusion_matrix.png")


# 7. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

misclassified, accuracy = evaluate_model("K-Nearest Neighbors", y_test, y_pred)
results.append(["K-Nearest Neighbors", misclassified, accuracy])

plot_decision_regions(X_combined_std, y_combined, knn, test_indices)
plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.title("K-Nearest Neighbors")
plt.legend()
plt.savefig(os.path.join(plots_folder, "knn_decision_boundary.png"))
plt.close()

show_confusion_matrix("K-Nearest Neighbors", y_test, y_pred, "knn_confusion_matrix.png")


# Create the final comparison table
comparison_table = pd.DataFrame(
    results,
    columns=["Classifier", "Misclassified Examples", "Accuracy"]
)

comparison_table["Accuracy"] = comparison_table["Accuracy"].round(3)

print("\nFinal Classifier Comparison")
print(comparison_table.to_string(index=False))

comparison_table.to_csv("iris_classifier_comparison.csv", index=False)
print("\nComparison table saved as iris_classifier_comparison.csv")
print("All plots were saved in the homework3plots folder.")