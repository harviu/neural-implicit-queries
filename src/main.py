import igl # work around some env/packaging problems by loading this first

import sys, os, time, math
from functools import partial

import jax
import jax.numpy as jnp

from skimage import measure


# Imports from this project
from geometry import *
from utils import *
from kd_tree import *
import implicit_mlp_utils, extract_cell

def dense_recon():
    # Construct the regular grid
    with Timer("full recon"):
        grid_res = (2 ** n_mc_depth + 1, 2 ** n_mc_depth + 1, 2 ** n_mc_depth + 1)
        ax_coords = jnp.linspace(-1., 1., grid_res[0])
        ay_coords = jnp.linspace(-1., 1., grid_res[1])
        az_coords = jnp.linspace(-1., 1., grid_res[2])
        grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ay_coords, az_coords, indexing='ij')
        grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
        sdf_vals = evaluate_implicit_fun(implicit_func, params, grid, batch_process_size)
        sdf_vals = np.array(sdf_vals, copy=True)
        sdf_vals = sdf_vals.reshape(grid_res)
        # marching cubes
        # delta = 1 / (np.array(grid_res) - 1)
        # bbox_min = grid[0,:]
        # verts, faces, normals, values = measure.marching_cubes(sdf_vals, level=isovalue, spacing=delta)
        # verts = verts + bbox_min[None,:]
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
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    data_bound = 1
    isovalue = 0

    data_opts = ['vorts', 'asteroid', 'combustion', 'ethanediol','isotropic']
    data_type = data_opts[4]
    if data_type == 'combustion':
        test_model = 'sample_inputs/jet_cz_elu_5_256.npz'
        input_file = '../data/jet_chi_0054.dat'
        bounds = np.array([479, 339, 119])
        isovalue = 1
    elif data_type == 'vorts':
        # test_model = 'sample_inputs/vorts_elu_5_128_l2.npz'
        # test_model = 'sample_inputs/vorts_relu_5_128.npz'
        test_model = 'sample_inputs/vorts_sin_5_128.npz'
        isovalue = 2
        input_file = '../data/vorts01.data'
        bounds = np.array([127, 127, 127])
    elif data_type == 'asteroid':
        # test_model = 'sample_inputs/v02_z_lr_elu_5_128.npz'
        test_model = 'sample_inputs/v02_z_sin_5_128.npz'
        input_file = '../data/99_500_v02.bin'
        bounds = np.array([499, 499, 499])
    elif data_type == 'ethanediol':
        test_model = 'sample_inputs/eth_elu_5_128.npz'
        input_file = '../data/ethanediol.bin'
        bounds = np.array([115, 116, 134])
    elif data_type == 'isotropic':
        test_model = 'sample_inputs/iso_sin_5_128.npz'
        input_file = '../data/Isotropic.nz'
        bounds = np.array([1024,1024,1024])

    # test_model = 'sample_inputs/bunny.npz'

    n_mc_depth = 10
    t = 0.95   # for sine function: 0.99999 will give all voxels
    n_mc_subcell= 3   #larger value may be useful for larger networks
    batch_process_size = 2 ** 12

    modes = ['affine_all', 'affine_truncate','uncertainty_all', 'uncertainty_truncate']
    mode = modes[2]
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 64
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = truncate_policies[0]

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode=mode, **affine_opts)
    print(params.keys())
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    # warm up
    print("== Warming up")
    indices = hierarchical(t)
    vals_np = dense_recon()

    # time
    print("== Test")
    indices = hierarchical(t)
    vals_np = dense_recon()

    # correctness
    if t<1:
        iso = iso_voxels(np.asarray(vals_np), isovalue)
        true_mask = np.zeros(((2 ** n_mc_depth + 1)**3,),np.bool_)
        true_mask[iso] = True
        our_mask = np.zeros(((2 ** n_mc_depth + 1)**3,),np.bool_)
        our_mask[indices] = True
        iou = (true_mask & our_mask).sum() / (true_mask | our_mask).sum()
        print('[iou]', iou)

    # compare tree
    num_level = (n_mc_depth - n_mc_subcell) * 3
    true_kd_array = kd_tree_array(implicit_func, params, num_level, dense=True, vals = vals_np)
    kd_array = kd_tree_array(implicit_func, params, num_level, prob_threshold=t, dense=False)
    # only the last level
    only_leaf = True
    if only_leaf == True:
        true_kd_array = true_kd_array[-2 ** num_level:]
        kd_array = kd_array[-2 ** num_level:]
    #IoU
    tree_iou = (true_kd_array & kd_array).sum() / (true_kd_array | kd_array).sum()
    print('[Tree IoU]', tree_iou)
    # F_score
    TP = (true_kd_array & kd_array).sum()
    FP = (~true_kd_array & kd_array).sum()
    FN = (true_kd_array & ~kd_array).sum()
    f_score = TP/(TP+(FP+FN)/2)
    print('[Tree F-score]', f_score)

    # save the binary files
    # data = load_data(data_type, input_file)
    # mean = data.mean()
    # std = data.std()
    # vals_np = (vals_np * std) + mean
    # save_vtk(vals_np.shape, bounds / vals_np.shape, vals_np, 'test.vti')
