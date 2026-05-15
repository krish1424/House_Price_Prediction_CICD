import pandas as pd
import joblib

from app.preprocess import preprocess_data
from app.model import train_model

df = pd.read_csv('data/housing_data.csv')

X, y, encoder = preprocess_data(df)

model = train_model(X, y)

joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(encoder, 'models/location_encoder.pkl')

print("Model Trained Successfully")