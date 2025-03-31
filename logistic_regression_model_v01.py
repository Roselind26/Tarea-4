# train_model.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Cargar datos
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Entrenar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Guardar modelo entrenado
joblib.dump(model, "logistic_regression_model_v01.pkl")

# (opcional) guardar también el scaler si lo usás luego
joblib.dump(scaler, "scaler.pkl")

print("✅ Modelo entrenado y guardado como logistic_regression_model_v01.pkl")
