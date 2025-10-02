# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import torch
import plotly.graph_objects as go
from pathlib import Path
import pickle
import io

# -----------------------------------------------------------------------------
#  THE DEFINITIVE FIX: A CUSTOM UNPICKLER FOR JOBLIB
#  This class tells joblib how to load PyTorch tensors. When it finds a tensor
#  that was saved on a GPU, it intercepts it and maps it to the CPU.
# -----------------------------------------------------------------------------
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_model_and_features():
    """
    Load the model and feature names using a custom unpickler to handle
    GPU-to-CPU model conversion seamlessly.
    """
    model_path = Path('model_compatible.pkl')
    features_path = Path('feature_list.json')

    if not model_path.exists() or not features_path.exists():
        st.error(f"Error: Make sure `tabnet_exoplanet.pkl` and `feature_list.json` are in the repository.")
        return None, None

    try:
        # Load the model using the custom unpickler
        with open(model_path, 'rb') as f:
            model = CPU_Unpickler(f).load()
        st.success("Model loaded successfully in CPU mode.")

    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None, None

    # Load features
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
    except Exception as e:
        st.error(f"Failed to read feature list: {e}")
        return None, None

    return model, features

def get_user_input(features):
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
model, feature_names = load_model_and_features()

st.title("🪐 TabNet Exoplanet Classifier")
st.markdown("This app predicts if an exoplanet candidate is **CONFIRMED** or a **FALSE POSITIVE**.")

VIRSYS_URL = "https://www.spaceappschallenge.org/2025/find-a-team/virsys/"
st.sidebar.markdown("---")
st.sidebar.markdown(f"Developed by **Team [Virsys]({VIRSYS_URL})**")

if model and feature_names:
    inputs = get_user_input(feature_names)
    col1, col2 = st.columns([1, 1.2])

    with col1:
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
                    orientation='h', marker_color=['#2ECC71', '#E74C3C']
                ))
                fig_prob.update_layout(
                    title_text='Prediction Confidence', xaxis_title="Confidence",
                    height=250, margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    with col2:
        st.subheader("Model Feature Importance")
        try:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names, 'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            fig_imp = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h'))
            fig_imp.update_layout(
                title_text='Global Feature Importances', xaxis_title="Importance Score",
                height=500, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        except Exception as e:
            st.info(f"Feature importance not available for this model. Details: {e}")
