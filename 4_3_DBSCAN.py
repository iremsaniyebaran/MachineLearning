# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 14:33:30 2025

@author: irems
"""


from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=1000, factor= 0.5, noise=0.05, random_state=42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])

dbscan = DBSCAN(eps = 0.1, min_samples=5)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
plt.title("DBSCAN sonuclari")







































