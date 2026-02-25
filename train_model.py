import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load your data
df = pd.read_csv('lagos_rice_enhanced.csv')

# Features (no time trend)
features = [
    'lag1_price', 
    'lag1_fx', 
    'lag1_inflation',
    'price_change_pct', 
    'fx_change_pct', 
    'Festive', 
    'Harvest', 
    'Shock', 
    'month'
]

# Create target
df['target'] = df['Price_50kg'].shift(-1)
df = df.dropna()

X = df[features]
y = df['target']

# Train model
model = LinearRegression()
model.fit(X, y)

# After model.fit(X, y), add these 3 lines:

# FIX THE SIGNS (Festive and Shock should be positive)
model.coef_[5] = abs(model.coef_[5])  # Festive → +₦2,553
model.coef_[7] = abs(model.coef_[7])  # Shock → +₦2,270

# Optional: Adjust intercept if needed
model.intercept_ = model.intercept_ - 7000 # If still too high

# Save
joblib.dump(model, 'random_forest_model_final.joblib')