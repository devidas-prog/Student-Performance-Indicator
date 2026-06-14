import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Load Dataset

df = pd.read_csv("StudentsPerformance_3_lyst1729690388778 (1).csv")

print(df.head())

# Data Checks


print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

print("\nStatistics:")
print(df.describe())


# Exploratory Data Analysis

plt.figure(figsize=(8,5))
sns.histplot(df["math score"], kde=True)
plt.title("Distribution of Math Scores")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="gender", y="math score", data=df)
plt.title("Gender vs Math Score")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    x="test preparation course",
    y="math score",
    data=df
)
plt.title("Test Preparation vs Math Score")
plt.show()

plt.figure(figsize=(8,5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Matrix")
plt.show()


# Feature Selection
X = df.drop("math score", axis=1)
y = df["math score"]


# Preprocessing
categorical_features = X.select_dtypes(
    include="object"
).columns

preprocessor = ColumnTransformer(
    transformers=[
        ( "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features )
    ],
    remainder="passthrough"
)


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Models

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

best_model = None
best_score = -999

# Training & Evaluation
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    results[name] = r2

    print("\n========================")
    print(name)
    print("========================")
    print("R2 Score :", r2)
    print("MAE      :", mae)
    print("MSE      :", mse)
    print("RMSE     :", rmse)

    if r2 > best_score:
        best_score = r2
        best_model = pipeline


# Best Model
print("\nBest Model Score:", best_score)


# Save Model
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Model saved as best_model.pkl")
