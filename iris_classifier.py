from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and structure the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Split features and labels
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData successfully split into training and testing sets.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Train the decision tree
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
