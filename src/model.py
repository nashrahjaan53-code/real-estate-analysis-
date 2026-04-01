"""Real Estate Data & Model"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

class RealEstateDataGenerator:
    @staticmethod
    def generate_property_data(n=10000):
        np.random.seed(42)
        data = []
        
        neighborhoods = ['Downtown', 'Suburbs', 'Waterfront', 'Historic', 'Tech Hub']
        
        for i in range(n):
            neighborhood = np.random.choice(neighborhoods)
            lat = np.random.uniform(40, 41)  # Realistic coordinates
            lon = np.random.uniform(-74, -73)
            
            # Price factors
            sqft = max(800, np.random.normal(2000, 800))  # Minimum 800 sqft
            bedrooms = np.random.choice([1, 2, 3, 4, 5])
            bathrooms = bathroom_count = np.random.poisson(2) + 1
            year_built = np.random.randint(1950, 2024)
            lot_size = max(1000, np.random.lognormal(4, 1.5))  # Minimum 1000
            
            # Market factors
            neighborhood_multiplier = {'Downtown': 1.3, 'Waterfront': 1.5, 'Tech Hub': 1.4, 'Historic': 0.9, 'Suburbs': 0.8}[neighborhood]
            crime_rate = np.random.uniform(1, 10)
            school_rating = np.random.uniform(1, 10)
            
            # Base price
            base_price = 150000 + (sqft * 200) + (bedrooms * 50000) + (bathrooms * 40000)
            price = base_price * (1 + (2024 - year_built) * -0.01) * neighborhood_multiplier
            price *= (1 - crime_rate * 0.005) * (1 + school_rating * 0.02)
            price += np.random.normal(0, price * 0.1)  # Noise
            
            data.append({
                'property_id': f'PROP_{i:06d}',
                'price': max(50000, price),
                'sqft': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built,
                'lot_size': lot_size,
                'neighborhood': neighborhood,
                'latitude': lat,
                'longitude': lon,
                'crime_rate': round(crime_rate, 1),
                'school_rating': round(school_rating, 1)
            })
        
        return pd.DataFrame(data)

class RealEstateModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05)
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def train(self, df):
        X = df.drop(['property_id', 'price', 'latitude', 'longitude'], axis=1)
        X['neighborhood'] = pd.Categorical(X['neighborhood']).codes
        self.feature_cols = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        return joblib.load(path)
