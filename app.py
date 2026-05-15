import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/house_price_model.pkl')

# Load location encoder
encoder = joblib.load('models/location_encoder.pkl')

# Title
st.title("🏠 House Price Prediction System")

st.write("Enter house details below")

# Inputs
area = st.number_input("Area in Square Feet", min_value=500)

rooms = st.number_input("Number of Rooms", min_value=1)

location = st.selectbox(
    "Select Location",
    ['Baner', 'Kothrud', 'Wakad', 'Hinjewadi',
     'Kharadi', 'Andheri', 'Powai', 'Bandra', 'Juhu']
)

# Prediction Button
if st.button("Predict Price"):

    # Encode location
    location_encoded = encoder.transform([location])[0]

    # Create dataframe
    data = pd.DataFrame({
        'area_sqft': [area],
        'rooms': [rooms],
        'location': [location_encoded]
    })

    # Predict
    prediction = model.predict(data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:.2f} Lakhs")