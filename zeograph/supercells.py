"""This module provides functions to generate and compare lattices
    using transformation matrices and linear algebra.
"""

__author__ = "Daniel Schwalbe-Koda"
__version__ = "1.0"
__email__ = "dskoda [at] mit [dot] edu"
__date__ = "Oct 7, 2019"


import numpy as np
from itertools import product


CHUNK_SIZE = 1000


def gen_matrices(Nmax):
    """Generates all 3x3 matrix of integers with absolute value smaller
        or equal to Nmax. Corresponds to Eq. D10 of the SI

    Args:
        Nmax (int)

    Returns:
        matrices (iterable): tuple (m11, m12, m13, m21, m22, m23, m31, m32, m33)
    """

    return product(range(-Nmax, Nmax + 1), repeat=9)


def determinant(tup):
    """Calculates the determinant of a tuple (m11, m12, m13, m21, m22, m23, m31, m32, m33)

    Args:
        tup (tuple of ints)

    Returns:
        det (int)
    """
    
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = tup

    det = m11 * m22 * m33 + m12 * m23 * m31 \
        + m13 * m21 * m32 - m11 * m23 * m32 \
        - m12 * m21 * m33 - m13 * m22 * m31

    return det


def gen_matrices_with_dets(Nmax):
    """Generates a dict containing all matrices with determinant larger than 0

    Args:
        Nmax (int)

    Returns:
        det_dict (dict): {determinant: list of tuples}
    """

    matrices = list(gen_matrices(Nmax))
    dets = [determinant(tup) for tup in matrices]

    return matrices, dets


def determinant_dict(Nmax):
    """Generates a dict containing all matrices with determinant larger than 0

    Args:
        Nmax (int)

    Returns:
        det_dict (dict): {determinant: list of tuples}
    """

    matrices, dets = gen_matrices_with_dets(Nmax)
    
    mat_array = np.array(matrices)
    det_array = np.array(dets)
    
    det2matrices = {}

    positive_dets = [x for x in set(dets) if x > 0]

    for d in positive_dets:
        idx = np.where (det_array == d)[0]
        det2matrices[d] = mat_array[idx]

    return det2matrices


def best_scaling(nA, nB, dets_available=[]):
    """Computes the relaxed least common multiple of two numbers nA and nB.
        Eq. D7 of the SI

    Args:
        nA (int)
        nB (int)
        dets_available (list): determinants which are available to be calculated.
            If not specified, assumes all ints are available.
    
    Returns:
        p, q (int): determinants to make nA * p ~= nB * q
    """

    def gcd(a,b):
        """Compute the greatest common divisor of a and b"""
        while b > 0:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        """Compute the lowest common multiple of a and b"""
        return int(a * b / gcd(a, b)) 

    # scaling for each number
    common_multiple = lcm(nA, nB)
    mA = common_multiple // nA
    mB = common_multiple // nB

    # all dets available
    if len(dets_available) == 0:
        return mA, mB

    # mA and mB in dets available
    elif mA in dets_available and mB in dets_available:
        return mA, mB

    # approximate mA and mB by other integers
    else:
        n = len(dets_available)
        p = np.array(dets_available).reshape(-1, 1) * np.ones((n, n))
        q = np.array(dets_available) * np.ones((n, n))
        
        closest_ratios = np.abs(mA / mB - p / q)
        i, j = np.unravel_index(closest_ratios.argmin(), closest_ratios.shape)
        
        return dets_available[i], dets_available[j] 


def chunk_list(l, n):
    """
    Chunks the list `l` into fragments of size `n`.

    Args:
        l (list)
        n (int)

    Returns:
        chunk (generator of lists)
    """
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def compare_lattices(A, B, chunk_size=CHUNK_SIZE):
    """Calculates the discrepancies between the transformed lattice matrices A, B.
        The matrices are defined in Eqs. D1 and D2, and the comparison in Eq. D13.

    Args:
        A (np.array): (n, 3, 3) array containing the transformed lattice vectors
        B (np.array): (m, 3, 3) array containing the transformed lattice vectors

    Returns:
        min_i (int): index of the transformed lattice of A that minimizes the distance
            between the lattices
        min_j (int): index of the transformed lattice of B that minimizes the distance
            between the lattices
        min_distance (float): minimum distance between the lattices
    """
    
    # we are interested in the angles, so we normalize the vectors
    normA = A / np.linalg.norm(A, axis=1, keepdims=True)
    normB = B / np.linalg.norm(B, axis=1, keepdims=True)

    # get only the angles between different lattice vectors 
    anglesA = np.einsum('ixj,ixk->ijk', normA, normA)[:, [0, 0, 1], [1, 2, 2]]
    anglesB = np.einsum('ixj,ixk->ijk', normB, normB)[:, [0, 0, 1], [1, 2, 2]]

    # compare the lattices
    min_i = None
    min_j = None
    min_distance = np.inf

    # performing the matrix operations is memory-intense, so we chunk it
    # to avoid doing the tensorial operations all at once
    for i, a in enumerate(chunk_list(anglesA, chunk_size)):
        for j, b in enumerate(chunk_list(anglesB, chunk_size)):
            c = np.expand_dims(a, axis=1) - np.expand_dims(b, axis=0)
            c = np.linalg.norm(c.reshape(a.shape[0], b.shape[0], -1), axis=-1)
            min_c = np.min(c)

            # makes an action only if we found a combination that is better than
            # our best supercell matching
            if min_c < min_distance:
                min_distance = min_c
                whereA, whereB = np.where(c == min_c)

                # indices of the supercell matching
                min_i = i * chunk_size + whereA[0]
                min_j = j * chunk_size + whereB[0]

    return min_i, min_j, min_distance
