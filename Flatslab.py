import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle

st.write("""
# Concrete Compressive Strength Prediction App
This app predicts the **Concrete Compressive Strength (fck)** based on input parameters.
""")
st.write('---')

# Display an image (update the image file if needed)
image = Image.open('concrete.jpg')  # Change filename if required
st.image(image, use_column_width=True)

# (Optional) Display the statistical summary as reference
st.subheader('Statistical Summary of Input Predictors and Compressive Strength')
st.markdown("""
| **Predictor**              | **Minimum** | **Maximum** | **Average** | **Standard Deviation** | **Skewness** |
|----------------------------|-------------|-------------|-------------|------------------------|--------------|
| fck (MPa)                  | 7.17        | 81.9        | 43.62       | 13.77                  | 0.28         |
| Cement (kg/m³)             | 78.00       | 635.00      | 362.22      | 107.94                 | -0.43        |
| Water (kg/m³)              | 105.00      | 277.00      | 183.72      | 26.98                  | 0.36         |
| FA (kg/m³)                 | 0.00        | 1200.00     | 684.09      | 295.08                 | -0.93        |
| RFA (kg/m³)                | 0.00        | 1200.00     | 157.91      | 268.91                 | 1.69         |
| CA (kg/m³)                 | 0.00        | 1170.00     | 426.75      | 332.45                 | -0.02        |
| RCA (kg/m³)                | 0.00        | 1115.00     | 334.46      | 323.59                 | 0.53         |
| MA (kg/m³)                 | 0.00        | 449.00      | 153.74      | 109.58                 | 0.39         |
| CHA (kg/m³)                | 0.00        | 320.00      | 6.05        | 14.62                  | 16.92        |
""")

# Sidebar for user input
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    # Note: We use the average value as the default for each slider.
    cement = st.sidebar.slider('Cement (kg/m³)', 78.0, 635.0, 362.22)
    water = st.sidebar.slider('Water (kg/m³)', 105.0, 277.0, 183.72)
    FA = st.sidebar.slider('FA (kg/m³)', 0.0, 1200.0, 684.09)
    RFA = st.sidebar.slider('RFA (kg/m³)', 0.0, 1200.0, 157.91)
    CA = st.sidebar.slider('CA (kg/m³)', 0.0, 1170.0, 426.75)
    RCA = st.sidebar.slider('RCA (kg/m³)', 0.0, 1115.0, 334.46)
    MA = st.sidebar.slider('MA (kg/m³)', 0.0, 449.0, 153.74)
    CHA = st.sidebar.slider('CHA (kg/m³)', 0.0, 320.0, 6.05)
    
    # Create a DataFrame of the user inputs
    data_user = {
        'Cement': cement,
        'Water': water,
        'FA': FA,
        'RFA': RFA,
        'CA': CA,
        'RCA': RCA,
        'MA': MA,
        'CHA': CHA
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

input_df = get_input_features()

# Display user input in the main panel
st.header('Specified Input Parameters')
st.write(input_df)
st.write('---')

# Load the optimized GradientBoostingRegressor model from pickle file
with open('optimized_gbrt_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.header('Prediction of Concrete Compressive Strength (fck)')
prediction = model.predict(input_df)
st.write(f"Predicted Concrete Compressive Strength (fck): {prediction[0]:.2f} MPa")
st.write('---')
