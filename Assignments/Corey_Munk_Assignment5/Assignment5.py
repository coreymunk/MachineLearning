#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

#%%
# Problem 1a. Kmeans

# Load data
x1_vals = np.load('x1_vals.npy')
x1df = pd.DataFrame(x1_vals, columns=['f1', 'f2'])

#%%
# Scale data and conduct EDA
std_scaler = StandardScaler()
x1df_scaled = pd.DataFrame(std_scaler.fit_transform(x1df[['f1', 'f2']]), columns = ['f1', 'f2'])

plt.figure(figsize = (7,7))
plt.scatter(data=x1df_scaled, x = 'f1', y = 'f2')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('x1_vals Scatterplot')    
plt.show()

sse = np.empty((0,2))
for k in np.arange(1, 21, 1):
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(x1df_scaled)
    sse = np.vstack([sse, np.array([k, kmeans.inertia_])])

plt.figure(figsize = (7,7))
plt.plot(sse[:,0], sse[:,1], marker='o')
plt.xlabel('Cluster Number')
plt.ylabel('Sum Square Error (Intertia)')
plt.title('Elbow Method')    
plt.xticks(ticks= np.arange(1, 21, 1))
plt.show()

#%%
# Basic Kmeans clustering model
kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 'auto', random_state = 321)
kmeans.fit(x1df_scaled)
centers = kmeans.cluster_centers_

print('Cluster Centroids:')
print(centers)

print('Iternations to Converge:', kmeans.n_iter_)

print('Iternations to Converge:', kmeans.inertia_)

plt.figure(figsize= (7, 7))
plt.scatter(data = x1df_scaled, x = 'f1', y = 'f2', c = kmeans.labels_, alpha = 0.8)
for j in np.arange(len(centers)):
    plt.scatter(x= centers[j, 0], y= centers[j, 1], marker= 'o', s= 40)
    plt.scatter(x= centers[j, 0], y= centers[j, 1], marker= 'o', s= 50, edgecolors= 'black', facecolors= 'none')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Basic Kmeans Model')    

plt.show()

# %%
# Questions: 

# A.What method(s) did you use to identify an appropriate value for K? Why did you select this method? (5 pts)

    # I used a scatter plot of the two features in the x1_vals data set and also an elbow plot. These methods
    # allowed me to visualize and analyze the data respectively to discern the number of clusters.

# B.What value did you select for K? Does your EDA support this choice? (2 pts)

# C.How many iterations were required before your model converged? (2 pts)

# D.What were the values for each of your cluster centroids? (2 pts)

# E.What kmeans measure serves as a proxy for cluster coherence? What value did your model return? 
#   Discuss your interpretation of this value. (5 pts)

