#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram

#%%
# Problem 1a. Kmeans

# Load data
x1_vals = np.load('x1_vals.npy')
x1df = pd.DataFrame(x1_vals, columns=['f1', 'f2'])

#%%
# Scale data and conduct EDA
std_scaler = StandardScaler()
x1df_scaled = pd.DataFrame(std_scaler.fit_transform(x1df[['f1', 'f2']]), columns = ['f1', 'f2'])

# Elbow Plot & Silhouette Plot
sse_silhouette = np.empty((0,3))
for k in np.arange(2, 21, 1):
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(x1df_scaled)
    labels = kmeans.labels_

    sse_silhouette = np.vstack([sse_silhouette, 
                         np.array([k, kmeans.inertia_ , silhouette_score(x1df_scaled, labels)])])

fig, (ax1, ax2) = plt.subplots(nrows= 1, ncols= 2, figsize= (12, 6))

ax1.scatter(x= sse_silhouette[:, 0], y= sse_silhouette[:, 1])
ax1.plot(sse_silhouette[:, 0], sse_silhouette[:, 1])
ax1.grid()
ax1.set_xlabel('Cluster number $k$')
ax1.set_ylabel('SSE (Inertia)')
ax1.set_xticks(ticks= np.arange(2, 21, 1))

ax2.scatter(x= sse_silhouette[:, 0], y= sse_silhouette[:, 2])
ax2.plot(sse_silhouette[:, 0], sse_silhouette[:, 2])
ax2.grid()
ax2.set_xlabel('Cluster number $k$')
ax2.set_ylabel('Silhouette Score')
ax2.set_xticks(ticks= np.arange(2, 21, 1))
plt.show()

#%%
# Basic Kmeans clustering model
kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 10, random_state = 321)
kmeans.fit(x1df_scaled)
centers = kmeans.cluster_centers_

print('Cluster Centroids:')
print(centers)

print('Iterations to Converge:', kmeans.n_iter_)

print('Cluster Coherence (Intertia):', kmeans.inertia_)

plt.figure(figsize= (7, 7))
plt.scatter(data = x1df_scaled, x = 'f1', y = 'f2', c = kmeans.labels_, alpha = 0.8)
for j in np.arange(len(centers)):
    plt.scatter(x= centers[j, 0], y= centers[j, 1], marker= 'o', s= 40)
    plt.scatter(x= centers[j, 0], y= centers[j, 1], marker= 'o', s= 50, edgecolors= 'black', facecolors= 'none')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('Basic Kmeans Model')    

plt.show()

#%%
# Problem 1b. Silhouette Plot

# Range of values for k
range_n_clusters = [2, 3, 4, 5, 6]

plt.figure(figsize=(len(range_n_clusters) * 6, 10))

for i, n_clusters in enumerate(range_n_clusters):
    # Initialize the clusterer
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(x1df_scaled)

    # Compute the silhouette scores
    silhouette_avg = silhouette_score(x1df_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(x1df_scaled, cluster_labels)

    plt.subplot(1, len(range_n_clusters), i + 1)
    y_lower = 10

    for j in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
        ith_cluster_silhouette_values.sort()
        size_cluster_j = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_j

        color = plt.cm.get_cmap("Spectral")(float(j) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
        y_lower = y_upper + 10

    plt.title("Silhouette plot for {} clusters".format(n_clusters))
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xticks(np.arange(-0.1, 1.1, 0.2))
    plt.xlim([-0.1, 1.0])

plt.show()

#%%
# Problem 2. External Validation

X, y = make_blobs(n_samples = 1000, centers = 5, n_features = 2, random_state = 321)
df = pd.DataFrame(X, columns= ['x1', 'x2'])
df['label'] = y

kmeans_EV = KMeans(n_clusters = 5, init = 'random', n_init = 10, random_state = 321)
df['pred_label'] = kmeans_EV.fit_predict(df[['x1', 'x2']])

# Use Adjusted Rand Score to compare actual vs predicted labels
adj_rand = adjusted_rand_score(df['label'], df['pred_label'])
print('Adjusted Rand Score: ', adj_rand)

plt.figure(figsize= (10,5))
plt.subplot(1,2,1)
plt.scatter(data = df, x = 'x1', y = 'x2', c = df['label'], alpha = 0.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Actual Lables')    

plt.subplot(1,2,2)
plt.scatter(data = df, x = 'x1', y = 'x2', c = df['pred_label'], alpha = 0.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Predicted Labels')    

plt.show()

#%%
# Problem 3a. Interpreting SSE
#   See word document for sketches and centroid location estimates.

#%%
# Problem 3b. Local and Global Objective Functions
#   See word document for sketches and centroid location estimates.

#%%
# Problem 3c. Density clustering
#   See word document for sketches and centroid location estimates.

#%%
# Problem 3d. Entropy vs. SSE
#   See word document for sketches and centroid location estimates.

#%%
# Problem 4. Selecting an Appropriate Clustering Algorithm

# Load Data
x4_vals = np.load('x4_vals.npy')
x4df = pd.DataFrame(x4_vals, columns=['f1', 'f2'])

# Scale data and conduct EDA
std_scaler = StandardScaler()
x4df_scaled = pd.DataFrame(std_scaler.fit_transform(x4df[['f1', 'f2']]), columns = ['f1', 'f2'])

# Elbow Plot & Silhouette Plot
sse_silhouette = np.empty((0,3))
for k in np.arange(2, 21, 1):
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(x4df_scaled)
    labels = kmeans.labels_

    sse_silhouette = np.vstack([sse_silhouette, 
                         np.array([k, kmeans.inertia_ , silhouette_score(x4df_scaled, labels)])])

fig, (ax1, ax2) = plt.subplots(nrows= 1, ncols= 2, figsize= (12, 6))

ax1.scatter(x= sse_silhouette[:, 0], y= sse_silhouette[:, 1])
ax1.plot(sse_silhouette[:, 0], sse_silhouette[:, 1])
ax1.grid()
ax1.set_xlabel('Cluster number $k$')
ax1.set_ylabel('SSE (Inertia)')
ax1.set_xticks(ticks= np.arange(2, 21, 1))

ax2.scatter(x= sse_silhouette[:, 0], y= sse_silhouette[:, 2])
ax2.plot(sse_silhouette[:, 0], sse_silhouette[:, 2])
ax2.grid()
ax2.set_xlabel('Cluster number $k$')
ax2.set_ylabel('Silhouette Score')
ax2.set_xticks(ticks= np.arange(2, 21, 1))
plt.show()

#%%
# Kmeans
kmeans = KMeans(n_clusters = 3, random_state = 321)
kmeans_labels = kmeans.fit_predict(x4df_scaled)

plt.figure(figsize= (7, 7))
plt.scatter(data = x4df_scaled, x = 'f1', y = 'f2', c = kmeans_labels, alpha = 0.8)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('x4_vals Kmeans')    
plt.show()

#%%
# Gaussian
gauss = GaussianMixture(n_components = 3, random_state = 321)
gauss_labels = gauss.fit_predict(x4df_scaled) # Use AIC_BIC?????

plt.figure(figsize= (7, 7))
plt.scatter(data = x4df_scaled, x = 'f1', y = 'f2', c = gauss_labels, alpha = 0.8)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('x4_vals Gaussian Mixture')    
plt.show()
#%%
# DBSCAN

neighbors = 3
nbrs = NearestNeighbors(n_neighbors= neighbors + 1).fit(x4df_scaled)
distances, indices = nbrs.kneighbors(x4df_scaled)
distances= np.sort(distances, axis= 0)

opt_esp_minpts = np.empty((0, 2))
plt.figure(figsize= (6, 6))
for n in np.arange(1, neighbors + 1):
    
    lndf = np.linspace(start= distances[0, n], stop= distances[-1, n], num= int(distances.shape[0]))

    distA = np.concatenate([np.arange(1, distances.shape[0] + 1) / distances.shape[0], distances[:, n]]).reshape(2, -1).T
    distB = np.concatenate([np.arange(1, lndf.shape[0] + 1) / lndf.shape[0], lndf]).reshape(2, -1).T

    all2all = cdist(XA = distA, XB= distB)
    idx_opt = all2all.min(axis= 1).argmax()
    
    opt_esp_minpts = np.vstack([opt_esp_minpts, np.array([n, distances[idx_opt, n]])]) 
    
    plt.plot(distances[:, n], label= n)
    plt.scatter(x= idx_opt, y= distances[idx_opt, n])
    
plt.legend()
plt.xlabel('Data Index sorted by distance')
plt.ylabel('Distance to NN')
plt.show()

print(opt_esp_minpts) # The knee of nearest neighbor (line 3) suggests eps = 0.21318469

dbs = DBSCAN(min_samples = 4, eps = 0.21318469)
dbs_labels = dbs.fit_predict(x4df_scaled)

#print('Calinski Harabasz Score: ',calinski_harabasz_score(X= x4df_scaled[['f1', 'f2']], labels= dbs_labels))

plt.figure(figsize= (7, 7))
plt.scatter(data = x4df_scaled, x = 'f1', y = 'f2', c = dbs_labels, alpha = 0.8)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('x4_vals DBSCAN')    
plt.show()

#%%
# Agglomerative Hierarchical

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

agg_hier = AgglomerativeClustering(n_clusters = 3, compute_distances=True, linkage='complete')
plot_dendrogram(agg_hier.fit(x4df_scaled))

agg_hier_labels = agg_hier.fit_predict(x4df_scaled)

plt.figure(figsize= (7, 7))
plt.scatter(data = x4df_scaled, x = 'f1', y = 'f2', c = agg_hier_labels, alpha = 0.8)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('x4_vals Agglomerative Hierarchical')    
plt.show()
