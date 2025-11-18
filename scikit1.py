import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------a----------------
# 1) Create Dataset
# ------------------------------
data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Price': [50, 70, 90, 110, 130]   # in lakhs
}

df = pd.DataFrame(data)

# ------------------------------
# 2) Split Features & Target
# ------------------------------
X = df[['Area']]   # input
y = df['Price']    # output

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3) Train Linear Regression Model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# 4) Make Prediction
# ------------------------------
predictions = model.predict(X_test)

# ------------------------------
# 5) Print Results
# ------------------------------
print("Test Data (Area):")
print(X_test.values)

print("\nActual Prices:", y_test.values)
print("Predicted Prices:", predictions)

# Slope & Intercept
print("\nSlope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Example: Predict price for 2200 sqft
print("\nPredicted price for 2200 sq ft:",
      model.predict([[2200]])[0], "lakhs")