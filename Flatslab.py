import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

st.write("""
# Fiber Reinforced Polymer Flat Slab Shear
This app predicts the **Ultimate Shear Capacity of Fiber Reinforced Polymer Flat Slab Shear**!
""")
st.write('---')

image = Image.open(r'soil.jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"finalequtionsmars.csv")
req_col_names = ["A_cm2", "bo_mm", "bo_1_5_mm", "de", "fc_Mpa", "p_percent", "Er_Gpa", "Vu_kN"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('Data Information')
st.write(data.head())
st.write(data.isna().sum())
corr = data.corr()
st.write(corr)

X = data.iloc[:, :-1]  # Features - All columns but last
y = data.iloc[:, -1]  # Target - Last Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the GradientBoostingRegressor
model = GradientBoostingRegressor(learning_rate=0.5, n_estimators=100)
model.fit(X_train, y_train)

st.sidebar.header('Specify Input Parameters')
def get_input_features():
    A_cm2 = st.sidebar.slider('A_cm2', 6.25, 1587.50, 671.10)
    bo_mm = st.sidebar.slider('bo_mm', 280.00, 2470.00, 1496.90)
    bo_1_5_mm = st.sidebar.slider('bo_1_5_mm', 640.00, 4608.00, 2509.18)
    de = st.sidebar.slider('de', 36.00, 284.00, 127.89)
    fc_Mpa = st.sidebar.slider('fc_Mpa', 22.16, 179.00, 44.72)
    p_percent = st.sidebar.slider('p_percent', 0.13, 3.76, 0.94)
    Er_Gpa = st.sidebar.slider('Er_Gpa', 28.40, 230.00, 74.44)
    
    data_user = {
        'A_cm2': A_cm2,
        'bo_mm': bo_mm,
        'bo_1_5_mm': bo_1_5_mm,
        'de': de,
        'fc_Mpa': fc_Mpa,
        'p_percent': p_percent,
        'Er_Gpa': Er_Gpa
    }
    
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()

# Main Panel
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Reads in saved classification model
load_clf = pickle.load(open('new.pkl', 'rb'))

st.header('Prediction of Vu (kN)')
# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---') 
