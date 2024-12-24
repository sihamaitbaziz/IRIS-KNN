
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()

# Create a DataFrame for better visualization
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target  # Add target species

# Map target numbers to species names
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


print(data.head())


print(data.isnull().sum())


print(data.describe())


sns.pairplot(data, hue='species', markers=["o", "s", "D"])
plt.show()

# Define features (X) and target (y)
X = data.drop('species', axis=1)  # Features: Sepal/Petal Length and Width
y = data['species']  # Target: Species

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from matplotlib.colors import ListedColormap

# Define the feature pairs for 2D plotting 
X_plot = X.iloc[:, [2, 3]].values  # Petal length and width
y_plot = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values

# Train the model on 2 features for visualization
knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_plot, y_plot)

# Create a mesh grid
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict on the grid
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
plt.title('KNN Decision Boundary (Petal Features)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()