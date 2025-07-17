# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:54:59 2025

@author: irems
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,5),
                         subplot_kw = {"xticks":[], "yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(digits.target[i])
    
plt.show







