# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:13:40 2020

@author: BOLD I.T
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%matplotlib inline
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'g')
plt.show()

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
Kmean.cluster_centers_
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='g')
plt.scatter(-0.94665068, -0.97138368, s=200, c='r', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='b', marker='s')
plt.show()
Kmean.labels_
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)