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
def load_model_and_features():
    """
    Loads the model by first initializing a new TabNetClassifier structure
    and then loading the saved weights into it. This is the most robust method.
    """
    weights_path = Path('tabnet_network_weights.pth')
    features_path = Path('feature_list.json')

    if not weights_path.exists() or not features_path.exists():
        st.error("Error: Make sure `tabnet_network_weights.pth` and `feature_list.json` are in the repository.")
        return None, None

    # Load features first to get model dimensions
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
    except Exception as e:
        st.error(f"Failed to read feature list: {e}")
        return None, None

    try:
        # 1. Initialize a new, empty TabNetClassifier with the same structure
        #    NOTE: If you used different parameters during your original training,
        #    you MUST match them here. These are the defaults from your notebook.
        model = TabNetClassifier(
            n_steps=8, n_d=16, n_a=16, gamma=1.5,
            mask_type="entmax"
        )

        # 2. Load the saved weights into the empty structure
        #    map_location='cpu' ensures it loads on the CPU.
        model.network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        
        # Manually set the device for the wrapper
        model.device = 'cpu'

        st.success("Model weights loaded successfully into new structure.")

    except Exception as e:
        st.error(f"Failed to load the model weights. Error: {e}")
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
                fig_prob.update_layout(title_text='Prediction Confidence', height=250)
                st.plotly_chart(fig_prob, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    # The feature importance part is removed for now as it's part of the full
    # trained object, not just the network. We can add it back if needed, but
    # the priority is getting prediction to work.
    with col2:
        st.info("Feature importance analysis is unavailable in this robust loading mode.")
