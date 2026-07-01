import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# import streamlit as st
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

df = pd.read_csv("car_project\car_data.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())


df = df.dropna()
df["Car_Age"] = 2026 - df["Year"]
df = df.drop(columns=["Year"])

df = pd.get_dummies(df, drop_first=True)

print(df.head())
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(),
    "knN Regression": KNeighborsRegressor(n_neighbors=5),
}

results = {}
for name, model in models.items():
    print("\n🔹 Training:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )

    results[name] = {"MAE": mae, "R2": r2}

    print("MAE:", mae)
    print("R²:", r2)
    print("RMSE:", rmse)
    print("Cross Val MAE:", -cv_scores.mean())

    results_df = pd.DataFrame(results).T

    print("\n===== MODEL COMPARISON =====")
    print(results_df)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["Selling_Price"], kde=True, color="blue", bins=20)
plt.title("Κατανομή των Τιμών Πώλησης(Selling Price)")
plt.xlabel("Selling Price")
plt.ylabel("Συχνότητα(Πλήθος Αυτοκινήτων)")

plt.subplot(1, 2, 2)
sns.histplot(df["Kms_Driven"], kde=True, color="green", bins=20)
plt.title("Κατανομή των Χιλιομέτρων(Kms Driven)")
plt.xlabel("Kms Driven")
plt.ylabel("Συχνότητα(Πλήθος Αυτοκινήτων)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=df["Selling_Price"], color="plum")
plt.title("Box Plot της Τιμής Πώλησης (Selling Price)")
plt.xlabel("Selling Price(σε Lakhs)")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Present_Price"],y=df["Selling_Price"], color="crimson", alpha=0.7)
plt.title("Σχἐση μεταξύ Present Price και Selling Price")
plt.xlabel("Present Price(Αρχική τιμή)")
plt.ylabel("Selling Price(Τιμή Πὠλησης)")
plt.show()


plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="Blues_d")
plt.title("Σύγκριση Μοντέλων με βάση το R2 Score")
plt.xlabel("Μοντέλα")
plt.ylabel("R2 Score")
plt.ylim(0, 1.0)
plt.show()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
importance = importance.sort_values(by="Importance", ascending=False)
importance.to_csv(
    "importance.csv",
    index=False
)

print(importance)

print("\n===== AI INSIGHT =====")

top_feature = importance.iloc[0]["Feature"]

print(f"The most important factor affecting car prices is: {top_feature}")

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance.head(10)),
plt.title("Top 10 Most Importance Feature Random Forest")
plt.xlabel("Βαθμός Σημαντικότητας")
plt.ylabel("Χαρακτηριστικά")
plt.tight_layout()
plt.show()

predictions = rf.predict(X_test)

residuals = y_test - predictions

plt.scatter(predictions, residuals)

plt.axhline(y=0, linestyle="--")

plt.show()
print("Saving model...")
joblib.dump(rf, "Car_price.pkl")
print("Model saved successfully")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

loaded_model = joblib.load("Car_price.pkl")
