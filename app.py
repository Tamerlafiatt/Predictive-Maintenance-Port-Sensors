"""
app.py — Streamlit UI for maintenance time prediction
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load("random_forest_model.pkl")
encoder  = joblib.load("onehot_encoder.pkl")

EQUIPEMENTS = [
    "GRUE PORTUAIRE LHM400", "GRUE PORTUAIRE LHM200", "GRUE PORTUAIRE LHM120",
    "CHARIOT TEREX 10T N04", "CHARIOT TEREX 10T N02", "CHARIOT TEREX 10T N06",
    "SPREADEUR AUTOMATIQUE BROMMA N04", "SPREADEUR AUTOMATIQUE BROMMA N01",
    "SPREADEUR AUTOMATIQUE BROMMA N03", "stacker terex", "stacker kalmar", "stacker LINDE"
]
PROBLEMES = [
    "problem electrique", "problem arret moteur", "problem chaine",
    "problem roue", "problem fuite", "problem piston et huile"
]

def predict_hours(equipement, probleme):
    df_in = pd.DataFrame([[equipement, probleme]],
                         columns=["Libellé Equipement", "Titre D.I"])
    encoded = encoder.transform(df_in)
    return round(rf_model.predict(encoded)[0], 1)

st.title("🛠️ Prédiction du Temps de Réparation")
st.markdown("Estimez le temps de réparation d'un équipement portuaire.")

col1, col2 = st.columns(2)
with col1:
    equip = st.selectbox("Équipement", EQUIPEMENTS)
with col2:
    prob  = st.selectbox("Type de panne", PROBLEMES)

if st.button("Prédire"):
    result = predict_hours(equip, prob)
    st.success(f"⏱️ Temps estimé : **{result} heures**")
    st.info(f"Équipement : {equip} | Panne : {prob}")
