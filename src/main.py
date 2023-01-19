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
        grid_res = (150, 100, 100)
        ax_coords = jnp.linspace(-1., 1., grid_res[0])
        ay_coords = jnp.linspace(-1., 1., grid_res[1])
        az_coords = jnp.linspace(-1., 1., grid_res[2])
        grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ay_coords, az_coords, indexing='ij')
        grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
        sdf_vals = evaluate_implicit_fun(implicit_func, params, grid, batch_process_size)
        sdf_vals = np.array(sdf_vals, copy=True)
        sdf_vals = sdf_vals.reshape(grid_res)
        # marching cubes
        delta = 1 / (np.array(grid_res) - 1)
        bbox_min = grid[0,:]
        verts, faces, normals, values = measure.marching_cubes(sdf_vals, level=isovalue, spacing=delta)
        verts = verts + bbox_min[None,:]
        return sdf_vals

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
    test_model = 'sample_inputs/vorts_elu_5_128_l2.npz'
    isovalue = 0
    n_mc_depth = 7
    t = 0.95
    batch_process_size = 2 ** 12

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode='affine_all')
    # print(params)
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))
    n_mc_subcell=3
    sdf_vals = dense_recon()
    print(sdf_vals.dtype)
    print(sdf_vals.shape)
    print(np.isfortran(sdf_vals))
    sdf_vals.tofile('test.bin')

    # warm up
    # hierarchical(t)
    # dense_recon()

    # # time
    # indices = hierarchical(t)
    # vals_np = dense_recon()

    # # correctness
    # iso = iso_voxels(vals_np, isovalue)
    # iou = len(np.intersect1d(indices, iso, True)) / len(np.union1d(indices, iso))
    # print('[iou]', iou)
