import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Step 1: Download dataset
path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
print("Dataset downloaded to:", path)
print("Files in folder:", os.listdir(path))

# Step 2: Load CSV with correct encoding
csv_path = os.path.join(path, "IMDb Movies India.csv")
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Step 3: Show first few rows and columns
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns)

# Step 4: Clean dataset
df_clean = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']].dropna()

# Step 5: Define features (X) and target (y)
X = df_clean[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df_clean['Rating']

# Step 6: Preprocessing for categorical data
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
)

# Step 7: Create pipeline with preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 8: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train the model
model.fit(X_train, y_train)

# Step 10: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"📊 Mean Squared Error on test data: {mse:.3f}")
import matplotlib.pyplot as plt

# Create a scatter plot of actual vs predicted ratings
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.grid(True)
plt.show()
