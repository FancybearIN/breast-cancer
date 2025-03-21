# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ipywidgets import interact, fixed

# Load the dataset
df = pd.read_csv('breast-cancer.csv')

# Display the first few rows of the dataset
print(df.head())

# Drop the 'id' column as it is not needed for the analysis
df.drop(columns=['id'], inplace=True)

# Display the first few rows after dropping the 'id' column
print(df.head())

# Display the shape of the dataset
print(df.shape)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2, random_state=2)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(x_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(x_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to load the breast cancer dataset
def load_data():
    cancer = datasets.load_breast_cancer()
    return cancer

# Function to plot decision boundaries
def plot_decision_boundaries(n_neighbors, data, labels):
    h = .02  # Step size in the mesh
    cmap_light = ListedColormap(['orange', 'blue'])
    cmap_bold = ListedColormap(['darkorange', 'darkblue'])

    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(data, labels)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'2-Class classification (k = {n_neighbors})')
    plt.show()

# Load the breast cancer dataset
cancer = load_data()

# Use only the first two features and standardize them
X = StandardScaler().fit_transform(cancer.data[:, :2])
y = cancer.target

# Interactive widget to plot decision boundaries
interact(plot_decision_boundaries, n_neighbors=(1, 20), data=fixed(X), labels=fixed(y))

# Try different k values and print accuracy scores
print("Accuracy scores for different k values:")
for i in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(f"For k={i}, Accuracy score is: {accuracy_score(y_test, y_pred):.2f}")