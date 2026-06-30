import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Fake dataset (same logic as notebook)
# -------------------------
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "hours_training": np.random.randint(1, 11, n),
    "explosiveness": np.random.randint(1, 10, n),
    "strength": np.random.randint(1, 10, n)
})

df["100m_time"] = (
    15
    - df["hours_training"] * 0.25
    - df["explosiveness"] * 0.3
    - df["strength"] * 0.2
    + np.random.normal(0, 0.3, n)
)

# -------------------------
# Train model
# -------------------------
X = df[["hours_training", "explosiveness", "strength"]]
y = df["100m_time"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# UI
# -------------------------
st.title("🏃 Sprint Predictor AI")

st.write("Enter your training stats to predict your 100m time")

hours = st.slider("Hours of Training", 1, 10, 5)
explosiveness = st.slider("Explosiveness", 1, 10, 5)
strength = st.slider("Strength", 1, 10, 5)

if st.button("Predict Sprint Time"):
    prediction = model.predict([[hours, explosiveness, strength]])[0]
    prediction = round(float(prediction), 2)

    st.success(f"Predicted 100m Time: {prediction} seconds")
