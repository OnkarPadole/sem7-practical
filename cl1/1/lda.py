# pip install scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply LDA
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test a new flower
flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(flower)

print("Predicted Species:", iris.target_names[prediction[0]])
