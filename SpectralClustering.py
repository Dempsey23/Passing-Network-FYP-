from sklearn.cluster import SpectralClustering
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Simulated passing data (replace this with your actual passing data)
def spectral(matrix,graph,name):
    passing_data = matrix

    # Compute similarity matrix (e.g., based on passing frequency)
    similarity_matrix = passing_data

    G = graph # Add edges between players with non-zero pass values

    # Compute spectral clustering
    num_clusters = 3
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    clusters = spectral_clustering.fit_predict(similarity_matrix)

    # Assign colors to nodes based on clusters
    node_colors = ['r' if cluster == 0 else 'b' if cluster == 1 else 'g' if cluster == 2 else 'm' if cluster == 3  else 'c' for cluster in clusters]

    # Plot the passing network with nodes colored by clusters
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, cmap='gnuplot')
    plt.title('Passing Network with Spectral Clustering')
    plt.savefig(f"{name}.png")
    plt.show()
