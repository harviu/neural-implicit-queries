import igl # work around some env/packaging problems by loading this first

import time
import jax
import jax.numpy as jnp

from skimage import measure
import numpy as np
import argparse


# Imports from this project
from utils import evaluate_implicit_fun, Timer, load_data
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

def get_dense_res(grid_res, lower, upper):
    # Construct the regular grid
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
    tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell, warm_up=True, dry = dry, mc_time = True)
    tri_pos = dense_recon_with_hierarchical_mc(implicit_func, params, isovalue, n_mc_depth, n_mc_subcell, warm_up=False, dry = dry , mc_time = True)
    return None


def test_hierarchical():

    tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        batch_process_size = batch_process_size, t=t, warm_up=True, dry=dry, mc_time=True)
    tri_pos = hierarchical_marching_cubes(implicit_func, params, \
        isovalue, lower, upper, n_mc_depth, n_subcell_depth=n_mc_subcell, \
        batch_process_size = batch_process_size, t=t, warm_up=False, dry=dry, mc_time = True)
    return None



if __name__ == "__main__":
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    data_bound = 1
    isovalue = 0

    data_opts = ['vorts', 'asteroid', 'combustion', 'ethanediol','isotropic','fox', 'hammer','birdcage','bunny']
############################################
    parser = argparse.ArgumentParser()

    # network
    parser.add_argument("--data_type", "-t", type=int, default=3)
    parser.add_argument("--n_mc_depth", "-d", type=int, default=8)

    # Parse arguments
    args = parser.parse_args()
    data_type = args.data_type
    n_mc_depth = args.n_mc_depth
############################################
    if data_type == 0:
        # test_model = 'sample_inputs/vorts_elu_8_32.npz'
        # test_model = 'sample_inputs/vorts_relu_8_32.npz'
        test_model = 'sample_inputs/vorts_sin_8_32.npz'
        input_file = '../data/vorts01.data'
        bounds = np.array([127, 127, 127])
        isovalue = 2
    elif data_type == 1:
        # test_model = 'sample_inputs/v02_relu_8_32.npz'
        # test_model = 'sample_inputs/v02_elu_8_32.npz'
        test_model = 'sample_inputs/v02_sin_8_32.npz'
        input_file = '../data/99_500_v02.bin'
        bounds = np.array([499, 499, 499])
    elif data_type == 2:
        # test_model = 'sample_inputs/jet_cz_elu_5_128.npz'
        test_model = 'sample_inputs/jet_sin_8_32.npz'
        input_file = '../data/jet_chi_0054.dat'
        bounds = np.array([479, 339, 119])
    elif data_type == 3:
        # test_model = 'sample_inputs/eth_elu_8_32.npz'
        # test_model = 'sample_inputs/eth_relu_8_32.npz'
        test_model = 'sample_inputs/eth_sin_8_32.npz'
        input_file = '../data/ethanediol.bin'
        bounds = np.array([115, 116, 134])
        isovalue = -2.2
    elif data_type == 4:
        test_model = 'sample_inputs/iso_sin_3_128.npz'
        # test_model = 'sample_inputs/iso_sin_5_128.npz'
        # test_model = 'sample_inputs/iso_sin_5_256.npz'
        input_file = '../data/Isotropic.nc'
        bounds = np.array([1024,1024,1024])
        isovalue = 0
    elif data_type == 5:
        test_model = 'sample_inputs/fox_relu.npz'
    elif data_type == 6:
        test_model = 'sample_inputs/hammer_relu.npz'
    elif data_type == 7:
        test_model = 'sample_inputs/birdcage_occ_relu.npz'
    elif data_type == 8:
        test_model = 'sample_inputs/bunny_elu.npz'

    # test_model = 'sample_inputs/bunny.npz'

############################################
    dry = n_mc_depth > 10
############################################
    n_mc_subcell= 3  #larger value may be useful for larger networks
    batch_process_size = 2 ** 12
    only_leaf = True
    # t = 0.68
    # t = 0.95
    # t = 0.997
    # t = 0.9999
    # t = 1

    # modes = ['uncertainty_all', 'affine_ua', 'affine_all', 'affine_fixed', 'affine_truncate', 'affine_append']
    modes = ['uncertainty_all']
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 64
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = truncate_policies[0]

    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    print('[Data]', data_opts[data_type])
    print(f"[Max depth] {n_mc_depth}")
    print(f"[Subcell depth] {n_mc_subcell}")
    print('[Dry Run]', dry)


############################################
    for i, mode in enumerate(modes):
############################################
        if mode == 'uncertainty_all':
            t_range = [2,10]
        elif mode == 'affine_ua':
            t_range = [5]
        else:
            t_range = [6,7,8,9]
############################################
        for t in t_range:
            implicit_func, params = generate_implicit_from_file(test_model, mode=mode, **affine_opts)
            
            # Dense reconstruction time test
            # if i == 0:
            #     print()
            #     print("[Dense]")
            #     test_dense()
            
            print()
            print('[Mode]', mode)
            print('[Threshold]', t)

            # time
            print("== Test")
            test_hierarchical()

            # find active cells (dense and hierarchical)
            # with Timer('calculate IoU'):
            MAXDEPTH = 10
            TP = 0
            FP = 0
            N_union = 0
            FN = 0
            TN = 0
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
                # our_mask = true_mask & kd_mask
                # difference_mask = np.logical_xor(true_mask,our_mask)
                TP = (true_mask & kd_mask).sum() 
                TN = (~true_mask & ~kd_mask).sum()
                FN = (true_mask & ~kd_mask).sum()
                FP = (~true_mask & kd_mask).sum()
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
                            # our_mask = true_mask & kd_mask
                            TP += (true_mask & kd_mask).sum()
                            TN += (~true_mask & ~kd_mask).sum()
                            FN += (true_mask & ~kd_mask).sum()
                            FP += (~true_mask & kd_mask).sum()
            print()
            print('[Total true active voxel]', TP + FN)
            print('[False Negative (missed)]', FN)
            FPR = FP / (FP + TN)
            FNR = FN / (FN + TP)
            print('[FPR]', FPR)
            print('[FNR]', FNR)
            print('[IoU]', TP/(TP + FN + FP))

            # difference_mask = difference_mask.astype(np.int32)

            # save the recon
            # from vtkmodules import all as vtk
            # from vtkmodules.util import numpy_support
            
            # data = load_data(data_opts[data_type], input_file)
            # mean = data.mean()
            # std = data.std()
            # print(data.shape)

            # vals_save = get_dense_res(data.shape, lower=lower, upper=upper)
            # vals_save = vals_save.astype(np.float32)
            # vals_save = (vals_save * std) + mean
            # print(vals_save.shape)

            # vtk_data = vtk.vtkImageData()
            # vtk_data.SetSpacing(1,1,1)
            # vtk_data.SetOrigin(0,0,0)
            # # vtk_data.SetDimensions(2 ** n_mc_depth,2 ** n_mc_depth,2 ** n_mc_depth)
            # vtk_data.SetDimensions(data.shape[2], data.shape[1], data.shape[0])
            # data_array = numpy_support.numpy_to_vtk(data.flatten())
            # vals_array = numpy_support.numpy_to_vtk(vals_save.flatten())
            # # diff_array = numpy_support.numpy_to_vtk(difference_mask.flatten())
            # data_array.SetName('orig')
            # vals_array.SetName('values')
            # # diff_array.SetName('diff')
            # vtk_data.GetPointData().AddArray(data_array)
            # vtk_data.GetPointData().AddArray(vals_array)
            # # vtk_data.GetPointData().AddArray(diff_array)
            # writer = vtk.vtkXMLDataSetWriter()
            # writer.SetFileName('%s.vti'% data_opts[data_type])
            # writer.SetInputData(vtk_data)
            # writer.Write()

            # compare tree
            # with Timer('tree F-score'):
            #     num_level = (n_mc_depth - n_mc_subcell) * 3
            #     vals_np = get_dense_values(depth = n_mc_depth, lower=lower, upper=upper)
            #     true_kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, dense=True, vals = vals_np)
            #     kd_array = kd_tree_array(implicit_func, params, num_level, isovalue, prob_threshold=t, dense=False)
            #     # only the last level
            #     if only_leaf == True:
            #         true_kd_array = true_kd_array[-2 ** num_level:]
            #         kd_array = kd_array[-2 ** num_level:]
            #     TP = (true_kd_array & kd_array).sum()
            #     FP = (~true_kd_array & kd_array).sum()
            #     FN = (true_kd_array & ~kd_array).sum()
            #     TN = (~true_kd_array & ~kd_array).sum()
            #     f_score = TP/(TP+(FP+FN)/2)
            #     # ppv = TP / (TP + FP)
            #     FPR = FP / (FP + TN)
            #     FNR = FN / (FN + TP)
            #     print()
            #     print('[Total Leaf]', 2**num_level)
            #     print('[True Number Leaf]', true_kd_array.sum())
            #     print('[Pred Number Leaf]', kd_array.sum())
            #     print('[Tree F-score]', f_score)
            #     print('[Tree FPR]', FPR)
            #     print('[Tree FNR]', FNR)
            #     # the F-sore if we predict every thing as active cell
            #     # print('[Dense F-score]', true_kd_array.sum() / (true_kd_array.sum() + (len(true_kd_array)-true_kd_array.sum())/2))
            #     # print('[PPV]', ppv)

            print('=========================')


