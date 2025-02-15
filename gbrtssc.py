import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle

# App Title and Description
st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Compressive Strength (fck in MPa)** of concrete using various mix design parameters!
""")
st.write('---')

# Image for context (replace with your own image relevant to concrete)
image = Image.open(r'soil.jpg')
st.image(image, use_column_width=True)

# Load your dataset with original column headers
# The dataset should have the following columns:
# fck (MPa), Cement (Kg/m3), Water (Kg/m3), FA(Kg/m3), RFA(Kg/m3), CA(kg/m3), RCA (Kg/m3), MA(Kg/m3), CHA(Kg/m3)
data = pd.read_csv(r"mars3 (1).csv")

# Data Information
st.subheader('Data Information')
st.write(data.head())
st.write("Missing values in each column:")
st.write(data.isna().sum())
st.write("Correlation Matrix:")
st.write(data.corr())

# Sidebar for input parameters
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    # Slider ranges are based on the provided statistical summary
    Cement = st.sidebar.slider('Cement (Kg/m3)', 78.00, 635.00, 362.22)
    Water  = st.sidebar.slider('Water (Kg/m3)', 105.00, 277.00, 183.72)
    FA     = st.sidebar.slider('FA(Kg/m3)', 0.00, 1200.00, 684.09)
    RFA    = st.sidebar.slider('RFA(Kg/m3)', 0.00, 1200.00, 157.91)
    CA     = st.sidebar.slider('CA(kg/m3)', 0.00, 1170.00, 426.75)
    RCA    = st.sidebar.slider('RCA (Kg/m3)', 0.00, 1115.00, 334.46)
    MA     = st.sidebar.slider('MA(Kg/m3)', 0.00, 449.00, 153.74)
    CHA    = st.sidebar.slider('CHA(Kg/m3)', 0.00, 320.00, 6.05)
    
    data_user = {
        'Cement (Kg/m3)': Cement,
        'Water (Kg/m3)': Water,
        'FA(Kg/m3)': FA,
        'RFA(Kg/m3)': RFA,
        'CA(kg/m3)': CA,
        'RCA (Kg/m3)': RCA,
        'MA(Kg/m3)': MA,
        'CHA(Kg/m3)': CHA
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

# Get input from the user
df = get_input_features()

# Display the input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Load your pre-trained model (ensure the file "optimized_gbrt_model (1).pkl" is in your working directory)
load_clf = pickle.load(open('optimized_gbrt_model (1).pkl', 'rb'))

# Predict the compressive strength based on input
st.header('Predicted Compressive Strength (MPa)')
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
