import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/game_sales_model.pkl")
columns = joblib.load("models/model_columns.pkl")

st.title("🎮 Video Game Sales Prediction System")

platforms = sorted([c.replace("platform_", "") for c in columns if c.startswith("platform_")])
genres = sorted([c.replace("genre_", "") for c in columns if c.startswith("genre_")])
publishers = sorted([c.replace("publisher_", "") for c in columns if c.startswith("publisher_")])

platform = st.selectbox("Platform", platforms)
genre = st.selectbox("Genre", genres)
publisher = st.selectbox("Publisher", publishers)

year = st.slider("Release Year", 1990, 2020, 2015)

na = st.number_input("NA Sales (millions)", 0.0, 50.0, 1.0)
eu = st.number_input("EU Sales (millions)", 0.0, 50.0, 1.0)
jp = st.number_input("JP Sales (millions)", 0.0, 50.0, 0.5)

# 1. Start with all-zero vector
input_df = pd.DataFrame(0, index=[0], columns=columns)

# 2. Fill numerical features
input_df["year"] = year
input_df["na_sales"] = na
input_df["eu_sales"] = eu
input_df["jp_sales"] = jp

# 3. Normalize text exactly like training
platform = platform.lower()
genre = genre.lower()
publisher = publisher.lower().replace(" ", "_")

# 4. Build correct one-hot column names
platform_col = f"platform_{platform}"
genre_col = f"genre_{genre}"
publisher_col = f"publisher_{publisher}"

# 5. Activate categorical columns safely
for col in [platform_col, genre_col, publisher_col]:
    if col in input_df.columns:
        input_df[col] = 1


if st.button("Predict Global Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Global Sales: {prediction:.2f} million units")
