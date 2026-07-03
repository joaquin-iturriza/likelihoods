#!/usr/bin/env python3.10

import numpy as np
import torch


def compute_invariants(particles, eps=1e-4, incl_diag_invariants=True):
    # compute matrix of all inner products
    def inner_product(p1, p2): return p1[..., 0] * p2[..., 0] - (
        p1[..., 1:] * p2[..., 1:]
    ).sum(dim=-1)
    if incl_diag_invariants:
        offset = 0
    else:
        offset = 1
    idxs = torch.triu_indices(
        particles.shape[-2], particles.shape[-2], offset=offset)
    invariants = inner_product(
        particles[..., idxs[0], :], particles[..., idxs[1], :])
    invariants = invariants.clamp(min=eps)
    return invariants.log()


def get_means_stds(particles):
    invs = compute_invariants(particles)
    return np.array(invs.mean(dim=-2)), np.array(invs.std(dim=-2))


for data in ['aag', 'aagg', 'zg', 'zgg', 'zggg', 'zgggg']:
    print(f'Calculating means and stds for {data}')
    particles = np.load(f'{data}.npy')
    shape = particles[:,:-1].shape
    particles = torch.tensor(particles[:,:-1].reshape(shape[0], -1, 4))
    means, stds = get_means_stds(particles)
    print(means)
    print(stds)
    np.save(f'{data}_means_stds.npy', (means, stds))