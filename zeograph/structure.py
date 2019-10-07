"""This module provides a class for describing and manipulating a
zeolite structure.
"""

__author__ = "Daniel Schwalbe-Koda"
__version__ = "1.0"
__email__ = "dskoda [at] mit [dot] edu"
__date__ = "Oct 7, 2019"


import numpy as np
import networkx as nx
from pymatgen.core import Structure


DEFAULT_RADIUS = 2


class Zeolite(Structure):

    def __init__(self, *args, **kwargs):
        super(Zeolite, self).__init__(*args, **kwargs)

        self._refresh()

    def _refresh(self):
        """Updates the crystal. Useful especially after creation of supercells
            or manipulation of the atomic coordinates.
        """

        self.silicon_atoms = self.get_species('Si')
        self.silicon_coords = np.array(
            [si.coords for si in self.silicon_atoms]
        )

        self.graph = self.get_cell_graph()

        return

    def create_supercell(self, *args, **kwargs):
        """Creates the supercell then updates the crystal.
        """

        super().create_supercell(*args, **kwargs)
        self._refresh()

        return

    def get_species(self, species):
        """Gets all sites in the unit cell corresponding to the given species.

        Args:
            species (str): element of the species to be extracted.

        Returns:
            atoms (list of sites)
        """

        atoms = [site for site in self.sites
                 if site.species_string == species]

        return atoms

    def get_expanded_si(self, radius=DEFAULT_RADIUS):
        """Gets all silicon sites with respect to oxygen sites.

        Args:
            radius (float): maximum radius when looking for neighbors.

        Returns:
            atoms (list of sites)
        """

        oxy_atoms = self.get_species('O')

        silicon = []
        for oxy in oxy_atoms:
            for nbr, _ in self.get_neighbors(oxy, radius):
                silicon.append(nbr)

        silicon = tuple(set(silicon))
        return silicon

    def get_adjacent_si(self, site, radius=DEFAULT_RADIUS):
        """Gets silicon sites adjacent to the given atom. If the given site
            is an oxygen, simply return its silicon neighbors. Otherwise, if
            it is a silicon, returns silicon atoms which are one oxygen away
            from it. Periodic boundary conditions are taken into account.

        Args:
            site (site): atom whose neighbors we want.
            radius (float): maximum radius when looking for neighbors.

        Returns:
            neighbors (list of sites): silicon atoms which are neighbors of
                the given site.
        """

        if site.species_string == 'Si':
            oxy_atoms = self.get_adjacent_oxy(site, radius=radius)

            neighbors = []
            for oxy in oxy_atoms:
                for nbr, _ in self.get_neighbors(oxy, radius):
                    if nbr not in neighbors and nbr != site:
                        neighbors.append(nbr)

        elif site.species_string == 'O':
            neighbors = [nbr for nbr, _ in self.get_neighbors(site, radius)
                         if nbr.species_string == 'Si']

        else:
            raise ValueError('site is neither a Si or O atom.')

        return neighbors

    def get_adjacent_oxy(self, silicon, radius=DEFAULT_RADIUS):
        """Gets all neighboring oxygen sites for the given silicon atom.
            Periodic boundary conditions are taken into account.

        Args:
            silicon (site): atom whose neighbors we want.
            radius (float): maximum radius when looking for neighbors.

        Returns:
            neighbors (list of sites): oxygen atoms which are neighbors of
                the given site.
        """

        neighbors = [oxy for oxy, _ in self.get_neighbors(silicon, radius)
                     if oxy.species_string == 'O']

        return neighbors

    def get_adj_matrix(self, radius=DEFAULT_RADIUS):
        """
        Creates an adjacency matrix based on the crystal structure.

        Args:
            radius (float): maximum radius when looking for neighbors.

        Returns:
            A (np.array): adjacency matrix of the crystal graph.
        """

        n = len(self.get_species('Si'))

        A = np.zeros((n, n), dtype=int)

        for oxygen in self.get_species('O'):
            neighbors = self.get_neighbors(oxygen, radius, include_index=True)

            neighbors = [x for x in neighbors
                         if x[0].species_string == 'Si']

            # element 2 is the index of the structure
            i = neighbors[0][2]
            j = neighbors[1][2]

            # multigraph adjacency matrix
            A[i, j] += 1
            A[j, i] += 1

        return A

    def get_periodic_graph(self, radius=DEFAULT_RADIUS):
        """
        Creates a zeolite graph based on the crystal structure.

        Args:
            radius (float): maximum radius when looking for neighbors.

        Returns:
            G (nx.MultiGraph): multigraph containing Si atoms as nodes and O
                bridges as edges. Periodic boundary conditions are taken
                into consideration.
        """

        G = nx.from_numpy_array(
            self.get_adj_matrix(radius=radius),
            parallel_edges=True,
            create_using=nx.MultiGraph()
        )

        return G

    def get_cell_graph(self, radius=DEFAULT_RADIUS):
        """
        Creates a zeolite graph based on the expanded crystal structure.

        Args:
            radius (float): maximum radius when looking for neighbors.

        Returns:
            G (nx.Graph): graph containing Si atoms as nodes and O
                bridges as edges. Periodic boundary conditions are NOT
                taken into account.
        """

        oxy_bridges = self.get_species('O')

        edgelist = []
        for oxy in oxy_bridges:
            nbr_list = [
                idx for nbr, _, idx, img in self.get_neighbors(
                    oxy, radius, include_index=True, include_image=True
                )
                if (
                    all(img == np.array([0, 0, 0])) and
                    nbr.species_string == 'Si'
                )

            ]

            if len(nbr_list) == 2:
                edgelist.append(nbr_list)

        G = nx.from_edgelist(edgelist, create_using=nx.Graph())

        return G
