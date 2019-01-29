import numpy as np
from numpy.random import binomial


def create_sampling_matrix_bernoulli(size):
    return [[(-1 if binomial(1, p=0.5) == 1 else 1) for i in range(size[0])] for j in range(size[1])]


def generate_matrix(siize = [10, 100]):
    mat = create_sampling_matrix_bernoulli(siize)
    np.save('bernoulli_' + str(siize[1]), mat)
    return mat


def load_matrix():
    return np.load('bernoulli.npy')
