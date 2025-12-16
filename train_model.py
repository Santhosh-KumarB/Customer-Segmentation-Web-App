import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Save the model
joblib.dump(kmeans, "customer_segmentation_model.pkl")

print("Model trained and saved successfully!")
