import numpy as np

def radial_basis(x, y, sigma_):
    dist = (x-y).T
    return np.exp(- (dist[0] ** 2 + dist[1] ** 2) / sigma_ ** 2)

def radial_basis_inv(x, y, sigma_):
    dist = (x-y).T
    return 1/np.sqrt(dist[0] ** 2 + dist[1] ** 2 + sigma_**2)

def radial_basis1(x, y, sigma_):
    dist = (x-y).T
    return np.exp(- (np.abs(dist[0]) + np.abs(dist[1])) / sigma_ )

def radial_basisM(x, y, sigma_):
    dist = (x-y).T
    return np.exp(- (dist[0] ** 2 + dist[1] ** 2) / sigma_ ** 2) * np.exp(- (np.abs(dist[0]) + np.abs(dist[1])) / sigma_ )