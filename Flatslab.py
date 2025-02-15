import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
from PIL import Image
import pickle

st.write("""
# Concrete Compressive Strength Prediction App
This app predicts the **Concrete Compressive Strength (fck)** based on input parameters!
""")
st.write('---')

# Display an image (update the image file as needed)
image = Image.open(r'soil.jpg')
st.image(image, use_column_width=True)

# Load your dataset
data = pd.read_csv(r"mars3 (1).csv")

# Define the required column names.
# The first column is the target (fck), and the rest are predictors.
req_col_names = ["fck", "Cement", "Water", "FA", "RFA", "CA", "RCA", "MA", "CHA"]
curr_col_names = list(data.columns)

# Create a mapper from current column names to required names if needed.
mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)

st.subheader('Data Information')
st.write(data.head())
st.write("Missing values:")
st.write(data.isna().sum())
st.write("Correlation Matrix:")
st.write(data.corr())

# Prepare features and target
X = data.drop("fck", axis=1)
y = data["fck"]

# Instead of training the model in the app, load the pre-trained model
with open('optimized_gbrt_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar for user input parameters (predictor values)
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    # Slider ranges based on your data summary:
    Cement = st.sidebar.slider('Cement (kg/m³)', 78.0, 635.0, 362.22)
    Water  = st.sidebar.slider('Water (kg/m³)', 105.0, 277.0, 183.72)
    FA     = st.sidebar.slider('FA (kg/m³)', 0.0, 1200.0, 684.09)
    RFA    = st.sidebar.slider('RFA (kg/m³)', 0.0, 1200.0, 157.91)
    CA     = st.sidebar.slider('CA (kg/m³)', 0.0, 1170.0, 426.75)
    RCA    = st.sidebar.slider('RCA (kg/m³)', 0.0, 1115.0, 334.46)
    MA     = st.sidebar.slider('MA (kg/m³)', 0.0, 449.0, 153.74)
    CHA    = st.sidebar.slider('CHA (kg/m³)', 0.0, 320.0, 6.05)
    
    data_user = {
        'Cement': Cement,
        'Water': Water,
        'FA': FA,
        'RFA': RFA,
        'CA': CA,
        'RCA': RCA,
        'MA': MA,
        'CHA': CHA
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()

# Main Panel: Display the input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Use the loaded model to make a prediction for fck (MPa)
st.header('Prediction of fck (MPa)')
prediction = model.predict(df)
st.write(f"Predicted Concrete Compressive Strength (fck): {prediction[0]:.2f} MPa")
st.write('---')
