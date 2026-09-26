#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:41:10 2026

@author: haoranzhu
"""

# -*- coding: utf-8 -*-
"""
ESG Clustering Analysis
@author: haoranzhu
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# =========================
# 1. Read Data
# =========================
df = pd.read_excel("ESG_Selected_Features_with_PCA.xlsx")

print(df.head())
print(df.columns)

# =========================
# 2. Keep original dataframe for interpretation
# =========================
df_original = df.copy()

# Drop text columns that should not be used in clustering
df_cluster = df.drop(columns=['SYMBOL', 'FULL_NAME', 'SECTOR'])

# If you do NOT want return to affect clustering, drop it too
# Because RETURN_1YR is more like an outcome variable
df_cluster = df_cluster.drop(columns=['RETURN_1YR'])

print(df_cluster.head())

# =========================
# 3. Standardize variables
# =========================
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

# =========================
# 4. Elbow Method for KMeans
# =========================
wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=47, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for ESG Clustering")
plt.show()

# =========================
# 5. KMeans Clustering
# =========================
kmeans = KMeans(n_clusters=3, random_state=47, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

df_original['KMeans_Cluster'] = clusters

# View cluster means
print("\nKMeans Cluster Summary:")
print(df_original.groupby('KMeans_Cluster')[
    ['ENV_SCORE', 'SOCIAL_SCORE', 'GOV_SCORE',
     'BETA', 'MARKET_CAP', 'EBITDA',
     'SECTOR_Real Estate', 'PC1', 'PC2', 'RETURN_1YR']
].mean())

# Count observations in each cluster
print("\nKMeans Cluster Counts:")
print(df_original['KMeans_Cluster'].value_counts())

pd.set_option('display.max_columns', None)
print(df_original.groupby('KMeans_Cluster')[
    ['ENV_SCORE', 'SOCIAL_SCORE', 'GOV_SCORE',
     'BETA', 'MARKET_CAP', 'EBITDA',
     'SECTOR_Real Estate', 'PC1', 'PC2', 'RETURN_1YR']
].mean())

# =========================
# 6. Visualize KMeans clusters using PCA axes
# =========================
plt.figure(figsize=(8,6))
plt.scatter(df_original['PC1'], df_original['PC2'], c=df_original['KMeans_Cluster'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters on PCA Dimensions")
plt.show()

# =========================
# 7. Dendrogram for Hierarchical Clustering
# =========================
plt.figure(figsize=(12,6))
dendrogram = sch.dendrogram(sch.linkage(df_scaled, method='ward'))
plt.title("Dendrogram for ESG Clustering")
plt.xlabel("Companies")
plt.ylabel("Distance")
plt.show()

# =========================
# 8. Hierarchical Clustering
# =========================
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_labels = hc.fit_predict(df_scaled)

df_original['HC_Cluster'] = hc_labels

print("\nHierarchical Cluster Summary:")
print(df_original.groupby('HC_Cluster')[
    ['ENV_SCORE', 'SOCIAL_SCORE', 'GOV_SCORE',
     'BETA', 'MARKET_CAP', 'EBITDA',
     'SECTOR_Real Estate', 'PC1', 'PC2', 'RETURN_1YR']
].mean())

print("\nHierarchical Cluster Counts:")
print(df_original['HC_Cluster'].value_counts())

# =========================
# 9. Visualize Hierarchical clusters using PCA axes
# =========================
plt.figure(figsize=(8,6))
plt.scatter(df_original['PC1'], df_original['PC2'], c=df_original['HC_Cluster'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clusters on PCA Dimensions")
plt.show()