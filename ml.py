import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("nmr_coffee_dataset.csv")

# Features and target
X = df.drop(columns=["origin"])
y = df["origin"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Random Forest
# ----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_preds))

# ----------------------------
# Train SVM
# ----------------------------
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("=== SVM Classification Report ===")
print(classification_report(y_test, svm_preds))

# ----------------------------
# Confusion Matrix Plot
# ----------------------------
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=rf.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

plot_conf_matrix(y_test, rf_preds, "Random Forest Confusion Matrix")
plot_conf_matrix(y_test, svm_preds, "SVM Confusion Matrix")
import joblib

# Save the Random Forest model
joblib.dump(rf, 'rf_coffee_classifier.pkl')

# Save the SVM model
joblib.dump(svm, 'svm_coffee_classifier.pkl')
joblib.dump(scaler, 'svm_scaler.pkl')

import json

# After evaluation
metrics = {
    "rf_accuracy": rf.score(X_test, y_test),
    "svm_accuracy": svm.score(X_test_scaled, y_test)
}

with open("models/model_metrics.json", "w") as f:
    json.dump(metrics, f)