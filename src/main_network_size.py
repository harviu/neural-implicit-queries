import igl # work around some env/packaging problems by loading this first

import time
import jax
import jax.numpy as jnp

from skimage import measure
import numpy as np


# Imports from this project
from utils import evaluate_implicit_fun, Timer
from evaluation import hierarchical_iso_voxels, kd_tree_array, iso_voxels
from kd_tree import hierarchical_marching_cubes, dense_recon_with_hierarchical_mc
from implicit_mlp_utils import generate_implicit_from_file

def get_dense_values(depth, lower, upper):
    # Construct the regular grid
    grid_res = (2 ** depth + 1, 2 ** depth + 1, 2 ** depth + 1)
    ax_coords = jnp.linspace(lower[0], upper[0], grid_res[0])
    ay_coords = jnp.linspace(lower[1], upper[1], grid_res[1])
    az_coords = jnp.linspace(lower[2], upper[2], grid_res[2])
    grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ay_coords, az_coords, indexing='ij')
    grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
    sdf_vals = evaluate_implicit_fun(implicit_func, params, grid, batch_process_size)
    sdf_vals = np.array(sdf_vals, copy=True)
    sdf_vals = sdf_vals.reshape(grid_res)
    # # marching cubes
    # delta = 1 / (np.array(grid_res) - 1)
    # bbox_min = grid[0,:]
    # verts, faces, normals, values = measure.marching_cubes(sdf_vals, level=isovalue, spacing=delta)
    # verts = verts + bbox_min[None,:]
    return sdf_vals

def test_dense():
    tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell, warm_up=True, dry = dry, mc_time = mc_time)
    tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell, warm_up=False, dry = dry , mc_time = mc_time)
    return None


def test_hierarchical():

    tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        batch_process_size = batch_process_size, t=t, warm_up=True, dry=dry, mc_time= mc_time)
    tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        batch_process_size = batch_process_size, t=t, warm_up=False, dry=dry, mc_time = mc_time)
    return None



if __name__ == "__main__":
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    data_bound = 1
    isovalue = 0
    mc_time = False

    data_opts = ['vorts', 'asteroid', 'combustion', 'ethanediol','isotropic','fox', 'hammer','birdcage','bunny']
############################################
    data_type = 4
    n_mc_depth = 10
############################################
    if data_type == 0:
        test_model = 'sample_inputs/vorts_sin_4_32.npz'
        input_file = '../data/vorts01.data'
        bounds = np.array([127, 127, 127])
        isovalue = 2
    elif data_type == 1:
        test_model = 'sample_inputs/v02_relu_8_32.npz'
        # test_model = 'sample_inputs/v02_elu_8_32.npz'
        # test_model = 'sample_inputs/v02_sin_8_32.npz'
        input_file = '../data/99_500_v02.bin'
        bounds = np.array([499, 499, 499])
    elif data_type == 2:
        # test_model = 'sample_inputs/jet_cz_elu_5_128.npz'
        test_model = 'sample_inputs/jet_sin_8_32.npz'
        input_file = '../data/jet_chi_0054.dat'
        bounds = np.array([479, 339, 119])
    elif data_type == 3:
        # test_model = 'sample_inputs/eth_sin_5_128.npz'
        test_model = 'sample_inputs/eth_sin_8_32.npz'
        input_file = '../data/ethanediol.bin'
        bounds = np.array([115, 116, 134])
        isovalue = -2.2
    elif data_type == 4:
        # test_model = 'sample_inputs/iso_sin_3_128.npz'
        test_model = 'sample_inputs/iso_sin_5_128.npz'
        # test_model = 'sample_inputs/iso_sin_5_256.npz'
        input_file = '../data/Isotropic.nz'
        bounds = np.array([1024,1024,1024])
        isovalue = 0
    elif data_type == 5:
        test_model = 'sample_inputs/fox.npz'
    elif data_type == 6:
        test_model = 'sample_inputs/hammer.npz'
    elif data_type == 7:
        test_model = 'sample_inputs/birdcage_occ.npz'
    elif data_type == 8:
        test_model = 'sample_inputs/bunny.npz'

    # test_model = 'sample_inputs/bunny.npz'

############################################
    dry = n_mc_depth > 10
############################################
    n_mc_subcell= 3  #larger value may be useful for larger networks
    batch_process_size = 2 ** 12

    modes = ['uncertainty_all', 'uncertainty_truncate', 'affine_ua', 'affine_all', 'affine_fixed', 'affine_truncate', 'affine_append']
    mode = modes[1]
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 1024
    affine_opts['affine_n_append'] = 512
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = truncate_policies[0]
    t = 5

    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    print('[Data]', data_opts[data_type])
    print(f"[Max depth] {n_mc_depth}")
    print(f"[Subcell depth] {n_mc_subcell}")
    print('[Dry Run]', dry)
    print('[Mode]', mode)
    print('[Threshold]', t)


############################################
    # for test_model in ['sample_inputs/vorts_sin_4_32.npz',
    #                    'sample_inputs/vorts_sin_8_32.npz',
    #                    'sample_inputs/vorts_sin_8_45.npz',
    #                    'sample_inputs/vorts_sin_8_64.npz',
    #                    'sample_inputs/vorts_sin_8_90.npz',
    #                    'sample_inputs/vorts_sin_8_128.npz',
    #                    'sample_inputs/vorts_sin_8_181.npz',
    #                    'sample_inputs/vorts_sin_8_256.npz',
    #                    'sample_inputs/vorts_sin_8_362.npz',
    #                    'sample_inputs/vorts_sin_8_512.npz',
    # ]:
############################################

    implicit_func, params = generate_implicit_from_file(test_model, mode=mode, **affine_opts)
    # Dense reconstruction time test

    print()
    print("[Dense]")
    test_dense()
    print()
    print("[Hierarchy]")
    test_hierarchical()

    #     # find active cells (dense and hierarchical)
    #     # with Timer('calculate IoU'):
    MAXDEPTH = 10
    N_intersection = 0
    N_union = 0
    N_missed = 0
    N_total = 0
    if n_mc_depth <= MAXDEPTH:
        vals_np = get_dense_values(depth = n_mc_depth, lower=lower, upper=upper)
        kd_mask = hierarchical_iso_voxels(implicit_func, params, \
            isovalue, lower, upper, n_mc_depth, n_mc_subcell, \
            batch_process_size = batch_process_size, t = t)
        # correctness
        iso = iso_voxels(np.asarray(vals_np), isovalue)
        true_mask = np.zeros(((2 ** n_mc_depth)**3,),np.bool_)
        true_mask[iso] = True
        true_mask = true_mask.reshape((2 ** n_mc_depth, )*3)
        our_mask = true_mask & kd_mask
        N_intersection = (true_mask & our_mask).sum() 
        N_union = (true_mask | our_mask).sum()
        N_missed = (true_mask & ~our_mask).sum()
        N_total = true_mask.sum()
        # print('[min,max]:', vals_np.min(), vals_np.max())
    else:
        n_subdivision = n_mc_depth - MAXDEPTH + 1
        for i in range(n_subdivision):
            for j in range(n_subdivision):
                for k in range(n_subdivision):
                    delta = (upper - lower) / n_subdivision
                    curr_lower = lower + delta * np.array([i,j,k])
                    curr_upper = lower + delta * (np.array([i,j,k]) + 1)
                    vals_np = get_dense_values(MAXDEPTH, curr_lower, curr_upper)
                    kd_mask = hierarchical_iso_voxels(implicit_func, params, \
                        isovalue, curr_lower, curr_upper, MAXDEPTH, n_mc_subcell, \
                        batch_process_size = batch_process_size, t = t)
                    iso = iso_voxels(np.asarray(vals_np), isovalue)
                    true_mask = np.zeros(((2 ** MAXDEPTH)**3,),np.bool_)
                    true_mask[iso] = True
                    true_mask = true_mask.reshape((2 ** MAXDEPTH, )*3)
                    our_mask = true_mask & kd_mask
                    N_intersection += (true_mask & our_mask).sum()
                    N_union += (true_mask | our_mask).sum()
                    N_missed += (true_mask & ~our_mask).sum()
                    N_total += true_mask.sum()
    print()
    print('[Total true active voxel]', N_total)
    print('[Number missed]', N_missed)
    print('[IoU]', N_intersection/N_union)

    # compare tree
    # with Timer('tree F-score'):
    # num_level = (n_mc_depth - n_mc_subcell) * 3
    # vals_np = get_dense_values(depth = n_mc_depth, lower=lower, upper=upper)
    # true_kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, dense=True, vals = vals_np)
    # kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, prob_threshold=t, dense=False)
    # true_kd_array = true_kd_array[-2 ** num_level:]
    # kd_array = kd_array[-2 ** num_level:]
    # TP = (true_kd_array & kd_array).sum()
    # FP = (~true_kd_array & kd_array).sum()
    # FN = (true_kd_array & ~kd_array).sum()
    # f_score = TP/(TP+(FP+FN)/2)
    # ppv = TP/(TP+FP)
    # print()
    # print('[Total Leaf]', 2**num_level)
    # print('[True Number Leaf]', true_kd_array.sum())
    # print('[Pred Number Leaf]', kd_array.sum())
    # print('[Tree F-score]', f_score)
    # # the F-sore if we predict every thing as active cell
    # print('[Dense F-score]', true_kd_array.sum() / (true_kd_array.sum() + (len(true_kd_array)-true_kd_array.sum())/2))
    # print('[PPV]', ppv)

    print('=========================')


    # save the binary files
    # data = load_data(data_type, input_file)
    # mean = data.mean()
    # std = data.std()
    # vals_np = (vals_np * std) + mean
    # save_vtk(vals_np.shape, bounds / vals_np.shape, vals_np, 'test.vti')
