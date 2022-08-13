"""
Provides methods for spectral clustering, with variable numbers of eigenvectors.
"""
import scipy as sp
import scipy.sparse.linalg
import scimagseg.imgraph
import scimagseg.objfunc
from sklearn.cluster import KMeans

def sc_precomputed_eigenvectors(eigvecs, num_clusters, num_eigenvectors):
    """
    Given an array of eigenvectors, run the k-means step of spectral clustering using the given number of eigenvectors.

    :param eigvecs: The precomputed eigenvectors
    :param num_clusters: The number of clusters to find
    :param num_eigenvectors: The number of eigenvectors to use for clustering
    :return: the found clusters
    """
    # Perform k-means on the eigenvectors to find the clusters
    labels = KMeans(n_clusters=num_clusters).fit_predict(eigvecs[:, :num_eigenvectors])

    # Split the clusters.
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters


def sc_cond(dataset: scimagseg.imgraph.DatasetGraph, num_clusters: int):
    """
    Given a dataset, and the desired number of clusters, run spectral clustering for every l < k, and return the
    clustering which optimises the conductance.
    """
    # First, compute all of the eigenvectors up front
    laplacian_matrix = dataset.graph.normalised_laplacian_matrix()
    _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, num_clusters, which='SM')

    # Optimal clustering
    best_clusters = None
    best_cond = None
    best_num_eigs = None

    for num_eigenvectors in range(1, num_clusters+1):
        found_clusters = scimagseg.sc.sc_precomputed_eigenvectors(eigvecs, num_clusters, num_eigenvectors)
        this_expansion = scimagseg.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
        if best_cond is None or scimagseg.objfunc.KWayExpansion.better(this_expansion, best_cond):
            best_cond = this_expansion
            best_clusters = found_clusters
            best_num_eigs = num_eigenvectors

    # Return the best clustering that we found
    print(f"Optimal number of eigenvectors: {best_num_eigs}")
    return best_clusters
