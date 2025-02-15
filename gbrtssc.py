import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pickle
from sklearn.preprocessing import StandardScaler

# App Title and Description
st.write("""
# Shear Capacity Prediction of Coupled Beams
This app predicts the **Shear Capacity of Coupled Beams** using various parameters!
""")
st.write('---')

# Image for context (replace with your own image)
image = Image.open(r'coupled_beam.jpg')
st.image(image, use_column_width=True)
# Load your dataset
data = pd.read_csv(r"angle beam.csv")  # Your dataset should contain the variables you listed
req_col_names = ["Ln", "bw", "h", "fc", "Ast", "fyt", "Ah", "fyh", "Av", "fyv", "Avd", "fyd", "Angle", "Vn"]
curr_col_names = list(data.columns)

# Rename columns to match required names
mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)

# Data Information
st.subheader('Data Information')
st.write(data.head())
st.write(data.isna().sum())
corr = data.corr()
st.write(corr)

# Sidebar for input parameters
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    Ln = st.sidebar.slider('L_n (mm)', 500.00, 1219.00, 823.37)
    bw = st.sidebar.slider('b_w (mm)', 120.00, 350.00, 221.11)
    h = st.sidebar.slider('h (mm)', 300.00, 991.00, 523.73)
    fc = st.sidebar.slider('fc (MPa)', 26.00, 85.00, 43.61)
    Ast = st.sidebar.slider('Ast (mm²)', 57.00, 2644.00, 518.99)
    fyt = st.sidebar.slider('fyt (MPa)', 276.00, 709.00, 458.36)
    Ah = st.sidebar.slider('Ah (mm²)', 0.00, 1058.00, 238.74)
    fyh = st.sidebar.slider('fyh (MPa)', 0.00, 614.00, 344.54)
    Av = st.sidebar.slider('Av (mm²)', 226.00, 5418.00, 1324.87)
    fyv = st.sidebar.slider('fyv (MPa)', 281.00, 953.00, 507.96)
    Avd = st.sidebar.slider('Avd (mm²)', 0.00, 2580.00, 829.36)
    fyd = st.sidebar.slider('Fyd (MPa)', 0.00, 883.00, 352.90)
    Angle = st.sidebar.slider('Angle (degrees)', 0.00, 40.60, 15.17)
    
    data_user = {
        'Ln': Ln,
        'bw': bw,
        'h': h,
        'fc': fc,
        'Ast': Ast,
        'fyt': fyt,
        'Ah': Ah,
        'fyh': fyh,
        'Av': Av,
        'fyv': fyv,
        'Avd': Avd,
        'fyd': fyd,
        'Angle': Angle
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

# Get input from the user
df = get_input_features()

# Standardize the input data
scaler = StandardScaler()
scaler.fit(data[req_col_names[:-1]])  # Exclude the target column 'Shear_Capacity'
df_standardized = scaler.transform(df)

# Convert standardized data back to DataFrame
df_standardized = pd.DataFrame(df_standardized, columns=df.columns)

# Display the standardized input parameters
st.header('Specified Input Parameters (Standardized)')
st.write(df_standardized)
st.write('---')

# Load your pre-trained model (make sure to have your model saved as a .pkl file)
load_clf = pickle.load(open('svr (2).pkl', 'rb'))

# Predict the shear capacity based on input
st.header('Predicted Shear Capacity (kN)')
prediction = load_clf.predict(df_standardized)
st.write(prediction)
st.write('---')
