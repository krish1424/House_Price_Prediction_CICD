import joblib
import pandas as pd

model = joblib.load('models/house_price_model.pkl')
encoder = joblib.load('models/location_encoder.pkl')

def predict_price(area, rooms, location):

    location_encoded = encoder.transform([location])[0]

    data = pd.DataFrame({
        'area_sqft': [area],
        'rooms': [rooms],
        'location': [location_encoded]
    })

    prediction = model.predict(data)

    return prediction[0]

if __name__ == "__main__":

    price = predict_price(1500, 3, 'Kharadi')

    print(f"Predicted Price: {price:.2f} Lakhs")