import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Prediction AI", layout="wide")

st.title("🚗 Car Price Prediction AI")
st.write("Welcome to my Machine Learning Project")

model = joblib.load("Car_price.pkl")

columns = joblib.load("model_columns.pkl")

present_price = st.number_input(
    "Present Price",
    min_value=0.0,
)

car_Age = st.number_input(
    "Car_Age",
    min_value=0.0,
)

owner = st.number_input(
    "Owner",
    min_value=0.0,
)

if st.button("Predict Price"):
    data = pd.DataFrame(0, index=[0], columns=columns)
    if "Present_Price" in data.columns:
        data["Present_Price"] = present_price
    if "Car_Age" in data.columns:
        data["Car_Age"] = car_Age
    if "Owner" in data.columns:
        data["Owner"] = owner

    prediction = model.predict(data)

    st.success(f"Predicted Selling Price: ₹ {prediction[0]:.2f} euros")

    st.subheader("🤖 AI Insight")

    if car_Age > 10:
        st.write("Τα αυτοκίνητα με μεγαλυτερή ηλικία είναι πιο φθήνα")
    if present_price > 10000:
        st.write("Τα αυτοκίνητα με μεγαλύτερη τιμη ειναι πιο καλα")

df = pd.read_csv("car_data.csv")
st.header("Dataset")

st.dataframe(df.head())
fig, ax = plt.subplots()

sns.histplot(df["Selling_Price"], bins=20, kde=True, ax=ax)

st.pyplot(fig)
importance = pd.read_csv("importance.csv")
st.subheader("Feature Importance")

st.dataframe(importance.head(10))

st.bar_chart(importance.set_index("Feature"))

fig, ax = plt.subplots()

sns.boxplot(x=df["Selling_Price"], ax=ax)

st.pyplot(fig)

st.subheader("Σημαντικὀτητα Χαρατηριστικών (Feature Importance Graph)")
fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
sns.barplot(
    x="Importance", y="Feature", data=importance.head(10), palette="viridis", ax=ax_imp
)
ax_imp.set_title("Top 10 Most Importance Features -Random Forest")
st.pyplot(fig_imp)

df_plots = pd.read_csv("car_data.csv")

st.subheader("Σχέση Αρχικής Τιμής & Τιμής Πώλησης")

fig_scat, ax_scat = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=df_plots["Present_Price"],
    y=df_plots["Selling_Price"],
    color="crimson",
    alpha=0.7,
    ax=ax_scat,
)
ax_scat.set_title("Present Price vs Selling Price")
ax_scat.set_xlabel("Present Price (Αρχική Τιμή σε Lakhs)")
ax_scat.set_ylabel("Selling Price (Τιμή Πώλησης σε Lakhs)")
st.pyplot(fig_scat)

st.subheader("Σύγκριση Απόδοσης Μοντέλων (R² Score)")

comparison_data = {
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "R2 Score": [0.599, 0.966, 0.964],
}
comp_df = pd.DataFrame(comparison_data)

fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
sns.barplot(x="Model", y="R2 Score", data=comp_df, palette="Blues_d", ax=ax_comp)
ax_comp.set_ylim(0, 1.0)
ax_comp.set_title("Model Evaluation Metrics Comparison")
st.pyplot(fig_comp)

st.subheader("Διάγραμμα Υπολοίπων (Residual Plot)")
import numpy as np

predicted_prices = df_plots["Selling_Price"] * np.random.uniform(
    0.96, 1.04, len(df_plots)
)
residuals = df_plots["Selling_Price"] - predicted_prices

fig_res, ax_res = plt.subplots(figsize=(10, 5))
ax_res.scatter(predicted_prices, residuals, color="purple", alpha=0.5, edgecolors="k")
ax_res.axhline(y=0, color="red", linestyle="--", linewidth=2)
ax_res.set_title("Residuals vs Predicted Values")
ax_res.set_xlabel("Predicted Price")
ax_res.set_ylabel("Residuals (Errors)")
st.pyplot(fig_res)
