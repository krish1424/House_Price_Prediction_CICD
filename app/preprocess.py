import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    encoder = LabelEncoder()

    df['location'] = encoder.fit_transform(df['location'])

    X = df[['area_sqft', 'rooms', 'location']]
    y = df['price']

    return X, y, encoder