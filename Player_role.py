import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

import TOT_VS_STOKE

# Example passing network data (you can replace this with your own data)


# Create a directed weighted graph for the passing network
G = TOT_VS_STOKE.G_TOT

# Compute the affinity matrix
n = len(G.nodes())
affinity_matrix = np.zeros((n, n))

for u, v, w in G.edges(data=True):
    affinity_matrix[u-1][v-1] = w['weight']  # Adjust for 0-based indexing

# Perform spectral clustering
n_clusters = 2  # Number of clusters (you can adjust this)
sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
sc.fit(affinity_matrix)

# Assign cluster labels to nodes
cluster_labels = sc.labels_

# Plot the passing network with colors representing clusters
pos = nx.spring_layout(G)  # Define layout for visualization
nx.draw(G, pos, node_color=cluster_labels, cmap=plt.cm.Set1, node_size=500, with_labels=True)
plt.title('Player Role Identification using Spectral Clustering')
plt.show()