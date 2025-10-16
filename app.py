# Importing dependencies
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# App title
st.title("🌍 Kenya CO₂ Emissions Forecast")

# Load dataset
kenya_df = pd.read_csv('kenya_co2.csv')

# Clean and prepare data
kenya_df.columns = kenya_df.columns.str.strip()
kenya_df.rename(columns={
    'Entity': 'Country',
    'Year': 'Year',
    'Annual CO₂ emissions (per capita)': 'CO2_per_capita'
}, inplace=True)

kenya_df['Year'] = pd.to_numeric(kenya_df['Year'], errors='coerce')
kenya_df['CO2_per_capita'] = pd.to_numeric(kenya_df['CO2_per_capita'], errors='coerce')
kenya_df = kenya_df.dropna()

# Split features and target
X = kenya_df[['Year']]
y = kenya_df['CO2_per_capita']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation of metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics on the app
st.subheader("📊 Model Evaluation")
st.write(f"**Mean Absolute Error:** {mae:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plotting results
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
ax.set_xlabel('Year')
ax.set_ylabel('CO₂ Emissions (per capita)')
ax.set_title('CO₂ Emissions Forecasting - Kenya')
ax.legend()
st.pyplot(fig)

# Forecast future emissions
st.subheader("📅 Forecast Future Emissions")
future_year = st.number_input("Enter a year to forecast:", min_value=2025, max_value=2100, value=2030)
future_pred = model.predict(np.array([[future_year]]))
st.write(f"**Forecasted CO₂ Emissions in {future_year}:** {future_pred[0]:.4f} metric tons per capita")
