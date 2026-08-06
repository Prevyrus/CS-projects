# Perceptron and Adaline: Decision Boundaries and Noise Robustness
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Create output folder

os.makedirs("outputs", exist_ok=True)

# Perceptron classifier
class Perceptron:
    def __init__(self, eta=0.1, n_iter=20, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0

        self.errors_ = []
        self.w_history_ = []
        self.b_history_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))

                self.w_ += update * xi
                self.b_ += update

                errors += int(update != 0.0)

            self.errors_.append(errors)

            # Save weights after each epoch for the animation
            self.w_history_.append(self.w_.copy())
            self.b_history_.append(self.b_)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Adaline Gradient Descent classifier

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=30, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0

        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = y - output

            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()

            loss = (errors ** 2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# Plot decision regions
def plot_decision_regions(X, y, classifier, ax, title, resolution=0.02):
    markers = ("o", "s")
    colors = ("red", "blue")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    labels = classifier.predict(grid_points)
    labels = labels.reshape(xx1.shape)

    ax.contourf(xx1, xx2, labels, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black"
        )

    ax.set_xlabel("Sepal length standardized")
    ax.set_ylabel("Petal length standardized")
    ax.set_title(title)
    ax.legend(loc="upper left")


# Load Iris dataset from local iris.data file
# Pull Iris dataset from UCI Machine Learning Repository if iris.data is not found
if not os.path.exists("iris.data"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    urllib.request.urlretrieve(url, "iris.data")
    print("Downloaded iris.data from UCI Machine Learning Repository.")
# The file iris.data should be in the same folder as the script.
df = pd.read_csv("iris.data", header=None, encoding="utf-8")

# Use only Setosa and Versicolor, first 100 examples.
# Column 0 = sepal length
# Column 2 = petal length
# Column 4 = class label
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)

X = df.iloc[0:100, [0, 2]].values

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Plot clean Iris data
plt.figure(figsize=(7, 5))

plt.scatter(
    X_std[y == 0, 0],
    X_std[y == 0, 1],
    color="red",
    marker="o",
    label="Setosa"
)

plt.scatter(
    X_std[y == 1, 0],
    X_std[y == 1, 1],
    color="blue",
    marker="s",
    label="Versicolor"
)

plt.xlabel("Sepal length standardized")
plt.ylabel("Petal length standardized")
plt.title("Clean Iris Data: Setosa vs Versicolor")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/01_clean_iris_data.png", dpi=300)
plt.show()


# Add noise by flipping 10% of class labels

rng = np.random.RandomState(42)

y_noisy = y.copy()

n_flip = int(0.10 * len(y_noisy))
flip_indices = rng.choice(len(y_noisy), size=n_flip, replace=False)

# Flip labels: 0 becomes 1, and 1 becomes 0
y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

print("Number of flipped labels:", n_flip)
print("Flipped indices:", flip_indices)



# Plot noisy Iris data
plt.scatter(
    X_std[y_noisy == 0, 0],
    X_std[y_noisy == 0, 1],
    color="red",
    marker="o",
    label="Class 0"
)

plt.scatter(
    X_std[y_noisy == 1, 0],
    X_std[y_noisy == 1, 1],
    color="blue",
    marker="s",
    label="Class 1"
)

plt.xlabel("Sepal length standardized")
plt.ylabel("Petal length standardized")
plt.title("Iris Data with 10% Flipped Labels")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/02_noisy_iris_data.png", dpi=300)
plt.show()


# Vary number of epochs

epoch_values = [1, 5, 10, 20]
epoch_errors = []

for n_epochs in epoch_values:
    ppn_temp = Perceptron(eta=0.1, n_iter=n_epochs, random_state=1)
    ppn_temp.fit(X_std, y)
    epoch_errors.append(ppn_temp.errors_[-1])

plt.plot(epoch_values, epoch_errors, marker="o")
plt.xlabel("Number of epochs")
plt.ylabel("Final number of updates")
plt.title("Effect of Varying Epochs on Perceptron Training")
plt.tight_layout()
plt.savefig("outputs/03_varying_epochs_perceptron.png", dpi=300)
plt.show()


# Train Perceptron and Adaline on clean and noisy data

ppn_clean = Perceptron(eta=0.1, n_iter=20, random_state=1)
ppn_clean.fit(X_std, y)

ppn_noisy = Perceptron(eta=0.1, n_iter=20, random_state=1)
ppn_noisy.fit(X_std, y_noisy)

ada_clean = AdalineGD(eta=0.01, n_iter=30, random_state=1)
ada_clean.fit(X_std, y)

ada_noisy = AdalineGD(eta=0.01, n_iter=30, random_state=1)
ada_noisy.fit(X_std, y_noisy)


# Accuracy comparison

ppn_clean_acc = accuracy_score(y, ppn_clean.predict(X_std))
ppn_noisy_acc = accuracy_score(y, ppn_noisy.predict(X_std))

ada_clean_acc = accuracy_score(y, ada_clean.predict(X_std))
ada_noisy_acc = accuracy_score(y, ada_noisy.predict(X_std))

print("\nAccuracy compared to original true labels:")
print(f"Perceptron clean: {ppn_clean_acc:.3f}")
print(f"Perceptron trained on noisy labels: {ppn_noisy_acc:.3f}")
print(f"Adaline clean: {ada_clean_acc:.3f}")
print(f"Adaline trained on noisy labels: {ada_noisy_acc:.3f}")


# Perceptron convergence plot: clean vs noisy

plt.plot(
    range(1, len(ppn_clean.errors_) + 1),
    ppn_clean.errors_,
    marker="o",
    label="Clean labels"
)

plt.plot(
    range(1, len(ppn_noisy.errors_) + 1),
    ppn_noisy.errors_,
    marker="s",
    label="Noisy labels"
)

plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.title("Perceptron Convergence: Clean vs Noisy Labels")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/04_perceptron_updates_clean_vs_noisy.png", dpi=300)
plt.show()


# Adaline loss plot: clean vs noisy

plt.plot(
    range(1, len(ada_clean.losses_) + 1),
    ada_clean.losses_,
    marker="o",
    label="Clean labels"
)

plt.plot(
    range(1, len(ada_noisy.losses_) + 1),
    ada_noisy.losses_,
    marker="s",
    label="Noisy labels"
)

plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
plt.title("Adaline Loss: Clean vs Noisy Labels")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/05_adaline_loss_clean_vs_noisy.png", dpi=300)
plt.show()


# Decision boundary comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_decision_regions(
    X_std,
    y,
    ppn_clean,
    axes[0, 0],
    "Perceptron: Clean Labels"
)

plot_decision_regions(
    X_std,
    y_noisy,
    ppn_noisy,
    axes[0, 1],
    "Perceptron: Noisy Labels"
)

plot_decision_regions(
    X_std,
    y,
    ada_clean,
    axes[1, 0],
    "Adaline: Clean Labels"
)

plot_decision_regions(
    X_std,
    y_noisy,
    ada_noisy,
    axes[1, 1],
    "Adaline: Noisy Labels"
)

plt.tight_layout()
plt.savefig("outputs/06_decision_regions_clean_vs_noisy.png", dpi=300)
plt.show()


# Accuracy bar chart
model_names = [
    "Perceptron\nClean",
    "Perceptron\nNoisy",
    "Adaline\nClean",
    "Adaline\nNoisy"
]

accuracies = [
    ppn_clean_acc,
    ppn_noisy_acc,
    ada_clean_acc,
    ada_noisy_acc
]

plt.bar(model_names, accuracies)
plt.ylabel("Accuracy against original labels")
plt.ylim(0, 1.05)
plt.title("Accuracy Comparison")
plt.tight_layout()
plt.savefig("outputs/07_accuracy_comparison.png", dpi=300)
plt.show()


# Animation: Perceptron decision boundary evolution
fig, ax = plt.subplots(figsize=(7, 5))

x_min = X_std[:, 0].min() - 1
x_max = X_std[:, 0].max() + 1
x_values = np.linspace(x_min, x_max, 100)


def animate(epoch):
    ax.clear()

    ax.scatter(
        X_std[y == 0, 0],
        X_std[y == 0, 1],
        color="red",
        marker="o",
        label="Setosa"
    )

    ax.scatter(
        X_std[y == 1, 0],
        X_std[y == 1, 1],
        color="blue",
        marker="s",
        label="Versicolor"
    )

    w = ppn_clean.w_history_[epoch]
    b = ppn_clean.b_history_[epoch]

    # Decision boundary:
    # w0*x + w1*y + b = 0
    # y = -(w0*x + b) / w1

    if abs(w[1]) > 1e-8:
        y_values = -(w[0] * x_values + b) / w[1]
        ax.plot(x_values, y_values, label="Decision boundary")

    ax.set_xlim(X_std[:, 0].min() - 1, X_std[:, 0].max() + 1)
    ax.set_ylim(X_std[:, 1].min() - 1, X_std[:, 1].max() + 1)

    ax.set_xlabel("Sepal length standardized")
    ax.set_ylabel("Petal length standardized")
    ax.set_title(f"Perceptron Decision Boundary Evolution - Epoch {epoch + 1}")
    ax.legend(loc="upper left")


animation = FuncAnimation(
    fig,
    animate,
    frames=len(ppn_clean.w_history_),
    interval=700,
    repeat=True
)

animation.save(
    "outputs/08_perceptron_decision_boundary_animation.gif",
    writer=PillowWriter(fps=2)
)

plt.close()


# Final message

print("\nDone. Files saved in the outputs folder:")
print("01_clean_iris_data.png")
print("02_noisy_iris_data.png")
print("03_varying_epochs_perceptron.png")
print("04_perceptron_updates_clean_vs_noisy.png")
print("05_adaline_loss_clean_vs_noisy.png")
print("06_decision_regions_clean_vs_noisy.png")
print("07_accuracy_comparison.png")
print("08_perceptron_decision_boundary_animation.gif")