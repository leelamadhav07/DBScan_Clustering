import streamlit as st

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
# SCALE
# ---------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------
# PAGE
# ---------------------------------

st.title("DBSCAN Clustering")

st.write("Density Based Customer Segmentation")

# ---------------------------------
# HYPERPARAMETERS
# ---------------------------------

eps = st.slider("EPS", 0.1, 2.0, 0.5)

min_samples = st.slider("Min Samples", 2, 20, 5)

# ---------------------------------
# MODEL
# ---------------------------------

model = DBSCAN(eps=eps, min_samples=min_samples)

clusters = model.fit_predict(X_scaled)

# ---------------------------------
# PLOT
# ---------------------------------

fig, ax = plt.subplots()

ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

ax.set_xlabel("Annual Income")

ax.set_ylabel("Spending Score")

ax.set_title("DBSCAN Clusters")

st.pyplot(fig)

# ---------------------------------
# NOISE COUNT
# ---------------------------------

noise = list(clusters).count(-1)

st.write(f"Noise Points: {noise}")
