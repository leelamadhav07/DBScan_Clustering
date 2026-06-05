import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------------
# LOAD DATA
# ---------------------------------

df = pd.read_csv("data/Mall_Customers.csv")

X = df.iloc[:, [3, 4]]

# ---------------------------------
# SCALING
# ---------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------
# DBSCAN
# ---------------------------------

model = DBSCAN(eps=0.5, min_samples=5)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

print(df.head())

print("Clusters Found:", len(set(clusters)))
