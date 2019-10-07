"""This module provides functions to calculate graph distances.
"""

__author__ = "Daniel Schwalbe-Koda"
__version__ = "1.0"
__email__ = "dskoda [at] mit [dot] edu"
__date__ = "Oct 7, 2019"

import numpy as np
import networkx as nx

EPS = 1e-10
WEIGHTS_DEFAULT = [0.45, 0.45, 0.10]

def distance_distribution(G):
    """Computes the distribution of node distances of a graph. Uses the
        Floyd-Warshall algorithm for the distances and formats it in
        the D-measure article standard.

    Args:
        G (nx.Graph): graph with N nodes

    Returns:
        nodes_distrib (np.array): (N, N) matrix containing the normalized
            distribution of distances of the graph. For each node i, the
            distribution is (p_1, p_2, ..., p_j, ..., p_N), where p_j is
            the proportion of nodes in the graph at distance j of the node i.
            Nodes with distance N are disconnected from the graph.
    """
    N = G.number_of_nodes()

    dist_matrix = np.asarray(nx.floyd_warshall_numpy(G, weight=1))
    dist_matrix[dist_matrix == np.inf] = N

    nodes_distrib = np.zeros((N, N + 1))
    for row in range(len(dist_matrix)):
        for length in dist_matrix[row]:
            nodes_distrib[row][int(length)] += 1

    nodes_distrib /= (N - 1)

    return nodes_distrib[:, 1:]


def distrib2diameter(mu):
    """Returns the diameter of the graph given its mean distribution
        of distances.

    Args:
        mu (np.array): mean distribution of distances

    Returns:
        diameter (int)
    """

    # Do not consider disconnected graphs
    mu = mu[:-1]
    return len(mu[mu > 0])


def entropy(p, eps=EPS):
    """Returns the entropy of a distribution.

    Args:
        p (np.array): normalized distribution.
        eps (float): threshold for numerical stability.

    Returns:
        entropy (float)
    """
    return -np.sum(p * np.log(np.maximum(p, eps)))


def NND(distributions_matrix, eps=EPS):
    """Returns the network node dispersion. Computed according to Eq. 1 in
        the original D-measure paper.

    Args:
        p (np.array): normalized distribution.
        eps (float): threshold for numerical stability.

    Returns:
        NND (float): network node dispersion
        mu_j (np.array): mean distribution of distances
    """
    N = distributions_matrix.shape[0]
    mu_j = np.mean(distributions_matrix, axis=0)

    J = np.max([0, entropy(mu_j) - entropy(distributions_matrix) / N])
    norm = np.log(np.max([2, distrib2diameter(mu_j) + 1]))

    return J / norm, mu_j


def alpha_centrality(G):
    """Computes the alpha-centrality of a graph.

    Args:
        G (nx.Graph)

    Returns:
        centrality (float)
    """
    N = G.number_of_nodes()

    degree = []
    for _, deg in nx.degree_centrality(G).items():
        degree.append(deg)
    degree = np.asmatrix(degree).T

    alpha = 1 / N

    A = nx.adjacency_matrix(G).todense()

    centrality = np.dot(np.linalg.inv(np.eye(N) - alpha * A.T), degree)
    centrality = np.sort(np.asarray(centrality).squeeze()) / (N**2)

    centrality = np.append(centrality, 1 - np.sum(centrality))

    return centrality


def zero_pad(mu_1, mu_2, end=True):
    """Zero pads one of the vectors so that both have the same length.
        Generally used to zero pad the mean distribution for a graph.

    Args:
        mu_1 (np.array)
        mu_2 (np.array)
        end (bool): if True, zero pads the end of the vector. To ensure
            compatibility with the D-measure calculation, the last element
            is kept as non-zero (infinite distance distribution weight).

    Returns:
        centrality (float)
    """
    L1 = len(mu_1)
    L2 = len(mu_2)

    if L1 > L2:
        z = np.zeros_like(mu_1)
        if end:
            z[:L2-1] = mu_2[:-1]
            z[-1] = mu_2[-1]
        else:
            z[-L2:] = mu_2
        mu_2 = z

    elif L2 > L1:
        z = np.zeros_like(mu_2)
        if end:
            z[:L1-1] = mu_1[:-1]
            z[-1] = mu_1[-1]
        else:
            z[-L1:] = mu_1
        mu_1 = z

    return mu_1, mu_2


def dmeasure(G1, G2, w=WEIGHTS_DEFAULT):
    """Calculates the D-measure between two graphs.

    Args:
        G1 (nx.Graph)
        G2 (nx.Graph)
        w (list of floats): weights w1, w2 and w3 from equation 2 of the
            original paper.

    Returns:
        D (float): D-measure between G1 and G2.
    """
    assert len(w) == 3, 'three weights have to be specified. Check argument `w`.'
    w1, w2, w3 = w

    # First term
    Pij_1 = distance_distribution(G1)
    Pij_2 = distance_distribution(G2)

    nnd_1, mu_1 = NND(Pij_1)
    nnd_2, mu_2 = NND(Pij_2)

    mu_1, mu_2 = zero_pad(mu_1, mu_2, end=True)

    mu_mean = (mu_1 + mu_2) / 2
    first = np.sqrt(
        np.maximum(
        (entropy(mu_mean) - (entropy(mu_1) + entropy(mu_2)) / 2), 0)
        / np.log(2)
    )

    # Second term
    second = np.abs(np.sqrt(nnd_1) - np.sqrt(nnd_2))

    # Third term
    alphaG_1 = alpha_centrality(G1)
    alphaG_2 = alpha_centrality(G2)
    alphaG_1, alphaG_2 = zero_pad(alphaG_1, alphaG_2, end=False)

    alphaG_mean = (alphaG_1 + alphaG_2) / 2
    third_1 = np.sqrt(
        np.maximum(
        (entropy(alphaG_mean) - (entropy(alphaG_1) + entropy(alphaG_2)) / 2), 0)
        / np.log(2)
    )

    # Complement
    alphaGcomp_1 = alpha_centrality(nx.complement(G1))
    alphaGcomp_2 = alpha_centrality(nx.complement(G2))
    alphaGcomp_1, alphaGcomp_2 = zero_pad(alphaGcomp_1,
                                          alphaGcomp_2, end=False)

    alphaGcomp_mean = (alphaGcomp_1 + alphaGcomp_2) / 2
    third_2 = np.sqrt(
        np.maximum((entropy(alphaGcomp_mean) - (entropy(alphaGcomp_1) + entropy(alphaGcomp_2)) / 2), 0)
        / np.log(2)
    )

    third = third_1 + third_2

    return w1 * first + w2 * second + w3 / 2 * third
