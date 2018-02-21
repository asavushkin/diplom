
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def p_transfrom():
    p=p_matrix("path_to_file",4)
    #p += p.T
    #Z = np.full((len(p[:,1]), len(p[:,1])), p[1][1])
    #W = Z - np.maximum( p, p.transpose())
    W = np.maximum( p, p.transpose())
    #print W
    return W 




p=p_transfrom()
#pd_matrix = pd.DataFrame(p)
#print pd_matrix

#p[np.isnan(p)] = 0
#print(p, "ЭТО МАТРИЦА")
X = StandardScaler().fit_transform(p)

##pd_matrix_X = pd.DataFrame(X)
print X

# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=2)
y_db = db.fit_predict(X)


#db = DBSCAN(eps=0.01, min_samples=1).p
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_

#print core_samples_mask,labels, "ЧТО ЭТО"
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


#clusters = [X[labels == i] for i in xrange(n_clusters_)]
print("______________",X[labels == 0],"___________")

distances_from_cluster  =  set(X[labels == 0][0])

#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))
# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

