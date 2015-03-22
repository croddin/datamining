import glob
import networkx as nx
import numpy as np
from sklearn.cluster import k_means

# Computes the combinatorial Laplacian L := D - A (hereafter simply the Laplacian) from the adjacency matrix of the given graph
def compute_laplacian(adj_matrix, weighted=False):
    # Computes the unweighted Laplacian from the possibly weighted adjacency matrix if so desired
    if not weighted:
        adj_matrix[adj_matrix > 0] = 1

    deg_matrix = np.diagflat(np.sum(adj_matrix, axis=0))
    
    return deg_matrix - adj_matrix       

# Computes the (sorted) spectrum from the Laplacian. This is a separate function since we should have this function automatically attempt to compute the eigenvalues in an alternative way if eigvalsh fails.
def compute_spectrum(laplacian):
    return np.sort(np.linalg.eigvalsh(laplacian))

# Something is not right with the graphs produced... if nothing else, the graphs have self-edges. So this function simply fixes that.
def fix_adj_matrix(orig_matrix):
    return orig_matrix - np.diagflat(np.diag(orig_matrix))

# Hack for making the adjacency matrix for a weighted graph from a nonnegative square matrix
def make_weight_graph(orig_matrix):
    return (orig_matrix + orig_matrix.T) / 2 - np.diagflat(np.ones(orig_matrix.shape[0]))

# Hack for making the adjacency matrix for an unweighted graph from a nonnegative square matrix: from the random entries of the first symmetric matrix produced, we assign an edge if that entry is >= 0.5.
def make_unweight_graph(orig_matrix):
    adj_matrix = (orig_matrix + orig_matrix.T) / 2 - np.diagflat(np.ones(orig_matrix.shape[0]))
    adj_matrix[adj_matrix >= 0.5 ] = 1
    adj_matrix[adj_matrix < 0.5 ] = 0
    return adj_matrix
    
if __name__ == '__main__':
    graph_paths = glob.glob("graphs/*/*.npz")
    test_graphs = []

    for graph_path in graph_paths:
        data = np.load(graph_path)
        test_graphs.append(fix_adj_matrix(data['connectivity']))

    n_graphs = len(test_graphs)
    
    # We subvert explicit for loops using list comprehensions.
    laplacians = [compute_laplacian(test_graphs[i], weighted=False) for i in range(n_graphs)] 
    spectra = np.array([compute_spectrum(laplacians[i]) for i in range(n_graphs)])

    # Run k-means with k-means++ initialization on spectra
    _, labels, _ = k_means(spectra, n_clusters=5)

    print labels

    # To add: HAC and other clustering algorithms, computing other graph invariants like the Cheeger constant.
    # To add: appropriately round eigenvalues, e.g., the smallest eigenvalue lambda_1 should always be zero.
