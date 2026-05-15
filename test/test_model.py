import pandas as pd
from app.model import train_model

def test_model_training():

    data = {
        'area_sqft': [600, 750, 900, 1200, 1300, 1600, 1800],
        'rooms': [1, 1, 2, 2, 3, 3, 4],
        'location': [1, 1, 2, 2, 3, 3, 4],
        'price': [30, 38, 45, 60, 72, 95, 110]
    }

    df = pd.DataFrame(data)

    X = df[['area_sqft', 'rooms', 'location']]
    y = df['price']

    model = train_model(X, y)

    prediction = model.predict([[1650, 3, 3]])

    assert prediction[0] > 0