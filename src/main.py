import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
from functools import partial

import jax
import jax.numpy as jnp

import argparse
import matplotlib
import matplotlib.pyplot as plt
import imageio
from skimage import measure


# Imports from this project
import render, geometry, queries
from geometry import *
from utils import *
import affine
import slope_interval
import sdf
import mlp
from kd_tree import *
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import implicit_mlp_utils, extract_cell
import affine_layers
import slope_interval_layers

def dense_recon():
    # Construct the regular grid
    with Timer("full recon"):
        grid_res = 2 ** n_mc_depth + 1
        ax_coords = jnp.linspace(-1., 1., grid_res)
        grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ax_coords, ax_coords, indexing='ij')
        grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
        # sdf_vals = jax.vmap(partial(implicit_func, params))(grid)
        sdf_vals = evaluate_implicit_fun(implicit_func, params, grid, batch_process_size)
        sdf_vals = sdf_vals.reshape(grid_res, grid_res, grid_res)
        # marching cubes
        delta = (grid[1,2] - grid[0,2]).item()
        bbox_min = grid[0,:]
        verts, faces, normals, values = measure.marching_cubes(np.array(sdf_vals), level=isovalue, spacing=(delta, delta, delta))
        verts = verts + bbox_min[None,:]
        return np.array(sdf_vals)

def hierarchical(t):

    print(f"do_hierarchical_mc {n_mc_depth}")

    with Timer("extract mesh"):
        # tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        #     isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        #     batch_process_size = batch_process_size)
        # tri_pos.block_until_ready()
        # tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
        # tri_pos = jnp.reshape(tri_pos, (-1,3))
        indices = hierarchical_iso_voxels(implicit_func, params, \
            isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
            batch_process_size = batch_process_size, t = t)
    return np.array(indices)

if __name__ == "__main__":
    data_bound = 1
    test_model = 'sample_inputs/vorts_sin_5_128.npz'
    isovalue = -0.5
    n_mc_depth = 8
    t = 1
    batch_process_size = 2 ** 12

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode='affine_all')
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))
    n_mc_subcell=3
    # warm up
    hierarchical(t)
    dense_recon()

    # time
    indices = hierarchical(t)
    vals_np = dense_recon()

    # correctness
    iso = iso_voxels(vals_np, isovalue)
    iou = len(np.intersect1d(indices, iso, True)) / len(np.union1d(indices, iso))
    print('[iou]', iou)
