# app.py
import streamlit as st
import numpy as np
import pandas as pd
import json
import torch
import plotly.graph_objects as go
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier

@st.cache_resource
def load_model_and_data():
    """
    Loads the model from its .zip archive and also loads the
    separately saved feature importance data.
    """
    model_archive_path = Path('tabnet_model.zip')
    features_path = Path('feature_list.json')
    importances_path = Path('feature_importances.json') # New file

    if not all([model_archive_path.exists(), features_path.exists(), importances_path.exists()]):
        st.error("Error: Make sure `tabnet_model.zip`, `feature_list.json`, and `feature_importances.json` are in the repository.")
        return None, None, None

    # Load model
    try:
        model = TabNetClassifier()
        model.load_model(model_archive_path)
        model.device = 'cpu'
        model.network.to(torch.device('cpu'))
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None, None, None

    # Load feature names
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
    except Exception as e:
        st.error(f"Failed to read feature list. Error: {e}")
        return None, None, None
        
    # Load feature importances
    try:
        with open(importances_path, 'r') as f:
            importance_data = json.load(f)
    except Exception as e:
        st.error(f"Failed to read feature importances. Error: {e}")
        return None, None, None

    st.success("Model and feature data loaded successfully.")
    return model, features, importance_data

def get_user_input(features):
    # This function remains the same
    st.sidebar.header('Input Exoplanet Features')
    user_inputs = {}
    for feature in features:
        min_val, max_val, default_val = 0.0, 1000.0, 50.0
        if "period" in feature: max_val, default_val = 500.0, 10.0
        elif "radius" in feature: max_val, default_val = 50.0, 5.0
        elif "temp" in feature: max_val, default_val = 3000.0, 800.0
        elif "flux" in feature: max_val = 10000.0
        elif "depth" in feature: max_val = 20000.0
        user_inputs[feature] = st.sidebar.slider(
            label=f'{feature.replace("_", " ").title()}',
            min_value=min_val, max_value=max_val, value=default_val, step=0.1
        )
    return user_inputs

st.set_page_config(page_title="Exoplanet Classifier", page_icon="🪐", layout="wide")
model, feature_names, importance_data = load_model_and_data()

st.title("🪐 TabNet Exoplanet Classifier")
st.markdown("This app predicts if an exoplanet candidate is **CONFIRMED** or a **FALSE POSITIVE**.")

VIRSYS_URL = "https://www.spaceappschallenge.org/2025/find-a-team/virsys/"
st.sidebar.markdown("---")
st.sidebar.markdown(f"Developed by **Team [Virsys]({VIRSYS_URL})**")

if model and feature_names and importance_data:
    inputs = get_user_input(feature_names)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        # Prediction logic remains the same
        st.subheader("Current Input Features")
        st.dataframe(pd.DataFrame([inputs]))
        if st.button('Predict', type="primary", use_container_width=True):
            input_array = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)
            try:
                probabilities = model.predict_proba(input_array)[0]
                prediction_index = np.argmax(probabilities)
                class_mapping = {0: 'CONFIRMED', 1: 'FALSE POSITIVE'}
                prediction_label = class_mapping[prediction_index]

                st.subheader("Prediction Result")
                if prediction_label == 'CONFIRMED':
                    st.success(f"**Result: CONFIRMED Exoplanet** 🎉")
                else:
                    st.warning(f"**Result: FALSE POSITIVE** ❌")

                fig_prob = go.Figure(go.Bar(
                    x=[f'{p:.1%}' for p in probabilities], y=['CONFIRMED', 'FALSE POSITIVE'],
                    orientation='h
