#python3 -m pip install pandas numpy scikit-learn xgboost
#brew install libomp
#python3 -m pip uninstall xgboost -y
#python3 -m pip install xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("Iris.csv")

# Features and target
X = df.drop(columns=["Id", "Species"], errors="ignore")
y = df["Species"]

# Convert labels into numbers
y = pd.factorize(y)[0]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=50, max_depth=3,
        eval_metric="mlogloss", random_state=42
    )
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_test, pred, average="weighted", zero_division=0))
    print("F1 Score:", f1_score(y_test, pred, average="weighted", zero_division=0))
