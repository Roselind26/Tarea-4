import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("model/breast_cancer_model.pkl")

st.title("Predicción de cáncer de mama")

st.markdown("Introduce los valores de las variables:")

# Definimos las variables de entrada (puedes ajustar según el modelo)
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean'
]

input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predecir"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    result = "Maligno" if prediction[0] == 0 else "Benigno"
    st.success(f"Resultado: {result}")
