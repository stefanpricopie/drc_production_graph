import math

import numpy as np
import networkx as nx


class MolErc:
    def __init__(self, name: str, x_name: list, x_ph: list):
        # Dataset Name
        self.name = name

        # search space & fitness values
        self.x_name = x_name                            # list of node names
        self.x_ph = x_ph                          # dataset of phenotypes - list of vectors
        # to be mapped
        self.nk = None
        self.x_fit = None

        # dag and ids of molecules in the dag
        self.dag = None
        self.pid = None         # ids of the molecules corresponding to the graph dag

    def append_dag(self, dag: nx.DiGraph):
        self.dag = dag

    # def __getattr__(self, k):   # is called automatically any time an unknown attribute is requested
    #     return getattr(self.dag, k)

    def map_nk(self, nk):
        if self.x_fit is not None:
            raise Exception('Fitness already calculated')
        else:
            self.nk = nk
            self.x_fit = list(map(nk.fit, self.x_ph))

    def __repr__(self):
        return "MolecularERC -> %s" % self.name


class NkLandscape:
    def __init__(self, n, k, seed=None):
        """create an NK landscape with given of dimension n and k
        n is the dimensionality, i.e. total number of genes, and
        k is the number of epistatic interactions per gene"""
        if k >= n:
            raise ValueError("k must be lower than n.")

        self.n = n
        self.k = k

        np.random.seed(seed)

        self.powersof2 = np.power(2, np.arange(self.k, -1, -1))  # used to find addresses on the landscape

        # interaction matrix for each component 0 to n-1 -> returns nxn binary matrix
        self.imatrix = self.imatrix_rand()
        # table of random U(0,1) numbers - contribution functions ci
        self.NK_land = np.random.rand(2**(self.k+1), self.n)

    def imatrix_rand(self):
        """
        This function takes the number of N elements and K interdependencies
        and creates a random interaction matrix.
        """
        int_matrix_rand = np.zeros((self.n, self.n))
        for aa1 in np.arange(self.n):
            Indexes_1 = list(range(self.n))
            Indexes_1.remove(aa1)  # remove self
            np.random.shuffle(Indexes_1)
            Indexes_1.append(aa1)   # re add self
            Chosen_ones = Indexes_1[-(self.k+1):]  # this takes the last K indexes and self(+1)
            for aa2 in Chosen_ones:
                int_matrix_rand[aa1, aa2] = 1  # we turn on the interactions with K other variables
        return int_matrix_rand.astype(bool)

    def fit(self, bitstring: str):
        """compute the fitness value for the given genome: the average of
        the individual fitness values of each gene in the genome"""
        genome = np.array([int(bit) for bit in bitstring])
        Fit_vector = np.zeros(self.n)
        for ad1 in np.arange(self.n):
            # select bits from the solution vector given by the interaction matrix
            try:
                NK_arg = genome[self.imatrix[ad1]]
            except IndexError as e:
                print(ad1, self.imatrix[ad1])
                print(genome)
                print(e)
            # convert bit or arguments for NK to index by multiplying with Power_key of 2
            Fit_vector[ad1] = self.NK_land[np.sum(NK_arg * self.powersof2), ad1]
        return np.mean(Fit_vector)

