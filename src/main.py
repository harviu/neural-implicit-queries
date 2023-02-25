import igl # work around some env/packaging problems by loading this first

import jax
import jax.numpy as jnp

from skimage import measure
import numpy as np


# Imports from this project
from utils import evaluate_implicit_fun, Timer
from evaluation import dense_recon_with_hierarchical_mc, hierarchical_iso_voxels, kd_tree_array, iso_voxels, compare_mc_clt
from kd_tree import hierarchical_marching_cubes
import implicit_mlp_utils

def get_dense_values():
    # Construct the regular grid
    grid_res = (2 ** n_mc_depth + 1, 2 ** n_mc_depth + 1, 2 ** n_mc_depth + 1)
    ax_coords = jnp.linspace(-1., 1., grid_res[0])
    ay_coords = jnp.linspace(-1., 1., grid_res[1])
    az_coords = jnp.linspace(-1., 1., grid_res[2])
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

def dense_recon():
    tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell)
    tri_pos.block_until_ready()
    tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
    tri_pos = jnp.reshape(tri_pos, (-1,3))
    with Timer("dense (GPU)"):
        tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell)
        tri_pos.block_until_ready()
        tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
        tri_pos = jnp.reshape(tri_pos, (-1,3))
    return None



def hierarchical():
    tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        batch_process_size = batch_process_size, t=t, warm_up=True)
    tri_pos.block_until_ready()
    tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
    tri_pos = jnp.reshape(tri_pos, (-1,3))
    with Timer("extract mesh"):
        # extract surfaces
        tri_pos = hierarchical_marching_cubes(implicit_func, params, \
            isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
            batch_process_size = batch_process_size, t=t)
        tri_pos.block_until_ready()
        tri_inds = jnp.reshape(jnp.arange(3*tri_pos.shape[0]), (-1,3))
        tri_pos = jnp.reshape(tri_pos, (-1,3))
        
    return None

if __name__ == "__main__":
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    data_bound = 1
    isovalue = 0

    data_opts = ['vorts', 'asteroid', 'combustion', 'ethanediol','isotropic','fox', 'hammer','birdcage','bunny']
    data_type = 0
    if data_type == 0:
        # test_model = 'sample_inputs/vorts_elu_5_128_l2.npz'
        # test_model = 'sample_inputs/vorts_relu_5_128.npz'
        test_model = 'sample_inputs/vorts_sin_8_32.npz'
        input_file = '../data/vorts01.data'
        bounds = np.array([127, 127, 127])
        isovalue = 2
    elif data_type == 1:
        # test_model = 'sample_inputs/v02_z_lr_elu_5_128.npz'
        test_model = 'sample_inputs/v02_z_sin_5_128.npz'
        input_file = '../data/99_500_v02.bin'
        bounds = np.array([499, 499, 499])
    elif data_type == 2:
        test_model = 'sample_inputs/jet_cz_elu_5_256.npz'
        # test_model = 'sample_inputs/jet_sin_3_128.npz'
        input_file = '../data/jet_chi_0054.dat'
        bounds = np.array([479, 339, 119])
    elif data_type == 3:
        # test_model = 'sample_inputs/eth_sin_5_128.npz'
        test_model = 'sample_inputs/eth_sin_8_32.npz'
        input_file = '../data/ethanediol.bin'
        bounds = np.array([115, 116, 134])
        isovalue = -2.2
    elif data_type == 4:
        test_model = 'sample_inputs/iso_sin_3_128.npz'
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

    n_mc_depth = 8
    # t = 0.68
    t = 0.95
    # t = 0.997
    # t = 0.9999
    # t = 1
    n_mc_subcell= 3  #larger value may be useful for larger networks
    batch_process_size = 2 ** 12
    evaluate = True
    only_leaf = True

    modes = ['affine_all', 'affine_truncate','uncertainty_all', 'uncertainty_truncate']
    mode = modes[2]
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 64
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = truncate_policies[0]

    print(f"[Max depth] {n_mc_depth}")
    print(f"[Subcell depth] {n_mc_subcell}")
    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode=mode, **affine_opts)
    print(params.keys())
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    # analyze histogram
    # for xx in np.linspace(-0.7,0.7, 10):
    #     from matplotlib import pyplot as plt
    #     from scipy.stats import norm
    #     plt.clf()

    #     center = jnp.array((xx,0,0))
    #     scale = jnp.array((0.3,0.3,0.3))
    #     range_lower = center - scale
    #     range_higher = center + scale

    #     mode = modes[2]
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode=mode, **affine_opts)
    #     vals, mu, sigma = compare_mc_clt(implicit_func, params, range_lower, range_higher)
    #     counts, bins = np.histogram(vals, 100,density=True)
    #     plt.stairs(counts, bins)
    #     x = np.linspace(mu - sigma * 4, mu+sigma*4, 100)
    #     y = norm.pdf(x, mu, sigma)
    #     # plt.xlim([x[0],x[-1]])
    #     plt.plot(x,y)
    #     # plt.savefig('hist_compare/%s_%s_x%.2f.png' % (data_opts[data_type],mode,xx))
        
    #     mode = modes[0]
    #     implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(test_model, mode=mode, **affine_opts)
    #     vals, mu, sigma = compare_mc_clt(implicit_func, params, range_lower, range_higher)
    #     x = np.linspace(mu - sigma * 4, mu+sigma*4, 100)
    #     y = norm.pdf(x, mu, sigma)
    #     plt.plot(x,y)

    #     plt.savefig('hist_compare/%s_x%.2f.png' % (data_opts[data_type],xx))


    # time
    print("== Test")
    hierarchical()
    dense_recon()

    if evaluate:
        # find active cells (dense and hierarchical)
        with Timer('calculate IoU'):
            vals_np = get_dense_values()
            print('[min,max]:', vals_np.min(), vals_np.max())
            kd_mask = hierarchical_iso_voxels(implicit_func, params, \
                isovalue, lower, upper, n_mc_depth, n_mc_subcell, \
                batch_process_size = batch_process_size, t = t)
            # correctness
            iso = iso_voxels(np.asarray(vals_np), isovalue)
            true_mask = np.zeros(((2 ** n_mc_depth)**3,),np.bool_)
            true_mask[iso] = True
            true_mask = true_mask.reshape((2 ** n_mc_depth, )*3)
            our_mask = true_mask & kd_mask
            iou = (true_mask & our_mask).sum() / (true_mask | our_mask).sum()
            print()
            print('[Total true active voxel]', true_mask.sum())
            print('[Number missed]', (true_mask & ~our_mask).sum())
            print('[IoU]', iou)

        # compare tree
        with Timer('tree F-score'):
            num_level = (n_mc_depth - n_mc_subcell) * 3
            true_kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, dense=True, vals = vals_np)
            kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, prob_threshold=t, dense=False)
            # only the last level
            if only_leaf == True:
                true_kd_array = true_kd_array[-2 ** num_level:]
                kd_array = kd_array[-2 ** num_level:]
            TP = (true_kd_array & kd_array).sum()
            FP = (~true_kd_array & kd_array).sum()
            FN = (true_kd_array & ~kd_array).sum()
            f_score = TP/(TP+(FP+FN)/2)
            print()
            print('[Total Leaf]', 2**num_level)
            print('[True Number Leaf]', true_kd_array.sum())
            print('[Pred Number Leaf]', kd_array.sum())
            print('[Tree F-score]', f_score)
            print('[Dense F-score]', TP / (TP + len(true_kd_array)/2))


    # save the binary files
    # data = load_data(data_type, input_file)
    # mean = data.mean()
    # std = data.std()
    # vals_np = (vals_np * std) + mean
    # save_vtk(vals_np.shape, bounds / vals_np.shape, vals_np, 'test.vti')
