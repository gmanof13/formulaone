import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np 

import matplotlib.pyplot as plt

con = duckdb.connect('formulaone.db')

df = con.execute("SELECT * FROM formulaone.main.formulaonedata").fetch_df()

df = df.drop(columns=['resultId', 'code', 'forename', 'surname', 'name', 
                     'nationality', 'number', 'position', 'rank', 
                     'raceId', 'round', 'status', 'laps', 'count_pit_stops'])

X = df.drop(columns=['points'])
y = df['points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# outliers 

plt.figure(figsize=(10, 6))

residuals = y_test - y_pred

threshold = 3 * np.std(residuals)

plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7, label='Predicted Points')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

outliers = np.abs(residuals) > threshold
plt.scatter(y_test[outliers], y_pred[outliers], color='orange', edgecolor='k', s=100, label='Outliers')

plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Predicted vs Actual Points with Outliers Highlighted')
plt.legend()
plt.show()

# Additional Visualizations

# 1. Histogram of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Line at zero
plt.show()

# 2. Box Plot of Residuals
plt.figure(figsize=(10, 6))
plt.boxplot(residuals, vert=False)
plt.title('Box Plot of Residuals')
plt.xlabel('Residuals')
plt.show()

# 3. Scatter Plot of Residuals vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', edgecolor='k', alpha=0.7)
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)  # Line at zero
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

