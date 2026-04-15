import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import joblib
import sklearn

# 1️⃣ Load dataset
df = pd.read_csv("water_potability.csv")  # keep relative path

# 2️⃣ Features & target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3️⃣ Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Compute imbalance weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 5️⃣ Create PIPELINE (🔥 important)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    ))
])

# 6️⃣ Train
pipeline.fit(X_train, y_train)

# 7️⃣ Predict
y_pred = pipeline.predict(X_test)

# 8️⃣ Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9️⃣ Create models folder safely
os.makedirs("models", exist_ok=True)

# 🔟 Save EVERYTHING in one file
joblib.dump({
    "model": pipeline,
    "sklearn_version": sklearn.__version__
}, "models/water_pipeline.pkl")

print("✅ Pipeline saved successfully!")