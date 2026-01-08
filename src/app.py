from utils import db_connect
engine = db_connect()

#-----------------------------------------
#Los IMPORTS
#-----------------------------------------
import streamlit as st         
import pandas as pd             
import joblib                   
from pathlib import Path        

#-----------------------------------------
#Ruta del modelo
#-----------------------------------------
#Ruta del modelo
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel desde src/
MODEL_PATH = BASE_DIR / "models" / "modelo_poliza.pkl"

#Cargamos el modelo
model = joblib.load(MODEL_PATH)

#Comprobamos si se ha cargado
st.success("Modelo cargado correctamente ✅")

# -----------------------------------------
# Nombramos la app para streamlit
# -----------------------------------------
st.title("Predicción de precio de póliza de seguros")

st.markdown("""
Ingrese los datos del cliente para predecir el precio estimado de la póliza.
""")

# -----------------------------------------
# Inputs del usuario
# -----------------------------------------
age = st.number_input("Edad", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoker_num = st.radio("¿Es fumador?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Sí")
smoker_yes = 1 if smoker_num == 1 else 0 

# -----------------------------------------
# Preparar los datos para el modelo
# -----------------------------------------
X_input = pd.DataFrame([[age, bmi, smoker_num, smoker_yes]], columns=model.feature_names_in_)

# -----------------------------------------
# Botón para "Simular""
# -----------------------------------------
if st.button("Simula ahora"):
    pred = model.predict(X_input)
    st.write(f"El precio de su póliza es de: {pred[0]:.2f} euros")
