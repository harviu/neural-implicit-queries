import igl # work around some env/packaging problems by loading this first

import time, math
import jax
import jax.numpy as jnp

from skimage import measure
import numpy as np
from functools import partial

# Imports from this project
import utils
from bucketing import *
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
from utils import Timer
from kd_tree import query_nodes
from implicit_mlp_utils import generate_implicit_from_file


@partial(jax.jit, static_argnames=("func","continue_splitting"), donate_argnums=(7,8,9,10))
def construct_uniform_unknown_levelset_tree_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper, out_n_valid,
        finished_interior_lower, finished_interior_upper, N_finished_interior,
        finished_exterior_lower, finished_exterior_upper, N_finished_exterior,
        isovalue=0.,offset=0.,prob_threshold=2,num_grid=1
        ):

    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]

    def eval_one_node(lower, upper):

        # perform an affine evaluation
        node_type = func.classify_box(params, lower, upper, isovalue=isovalue, offset=offset, prob_threshold=prob_threshold, num_grid=num_grid)

        # use the largest length along any dimension as the split policy
        worst_dim = jnp.argmax(upper-lower, axis=-1)

        return node_type, worst_dim
        
    # evaluate the function inside nodes
    node_types, node_split_dim = jax.vmap(eval_one_node)(node_lower, node_upper)

    # if requested, write out interior nodes
    if finished_interior_lower is not None:
        out_mask = jnp.logical_and(node_valid, node_types == SIGN_NEGATIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_interior
        finished_interior_lower = finished_interior_lower.at[out_inds,:].set(node_lower, mode='drop')
        finished_interior_upper = finished_interior_upper.at[out_inds,:].set(node_upper, mode='drop')
        N_finished_interior += jnp.sum(out_mask)
    
    # if requested, write out exterior nodes
    if finished_exterior_lower is not None:
        out_mask = jnp.logical_and(node_valid, node_types == SIGN_POSITIVE)
        out_inds = utils.enumerate_mask(out_mask) + N_finished_exterior
        finished_exterior_lower = finished_exterior_lower.at[out_inds,:].set(node_lower, mode='drop')
        finished_exterior_upper = finished_exterior_upper.at[out_inds,:].set(node_upper, mode='drop')
        N_finished_exterior += jnp.sum(out_mask)

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])
    N_new = jnp.sum(split_mask) # each split leads to two children (for a total of 2*N_new)

    ## now actually build the child nodes
    if continue_splitting:

        # extents of the new child nodes along each split dimension
        new_lower = node_lower
        new_upper = node_upper
        new_mid = 0.5 * (new_lower + new_upper)
        new_coord_mask = jnp.arange(3)[None,:] == node_split_dim[:,None]
        newA_lower = new_lower
        newA_upper = jnp.where(new_coord_mask, new_mid, new_upper)
        newB_lower = jnp.where(new_coord_mask, new_mid, new_lower)
        newB_upper = new_upper

        # concatenate the new children to form output arrays
        node_valid = jnp.concatenate((split_mask, split_mask))
        node_lower = jnp.concatenate((newA_lower, newB_lower))
        node_upper = jnp.concatenate((newA_upper, newB_upper))
        new_N_valid = 2*N_new
        outL = out_valid.shape[1]

    else:
        node_valid = jnp.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        new_N_valid = jnp.sum(node_valid)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid = out_valid.at[ib,:outL].set(node_valid)
    out_lower = out_lower.at[ib,:outL,:].set(node_lower)
    out_upper = out_upper.at[ib,:outL,:].set(node_upper)
    out_n_valid = out_n_valid + new_N_valid

    return out_valid, out_lower, out_upper, out_n_valid, \
           finished_interior_lower, finished_interior_upper, N_finished_interior, \
           finished_exterior_lower, finished_exterior_upper, N_finished_exterior


def construct_uniform_unknown_levelset_tree(func, params, lower, upper, node_terminate_thresh=None, split_depth=None, with_interior_nodes=False, with_exterior_nodes=False, isovalue=0., offset=0., batch_process_size=4096, prob_threshold = 1.):
       
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b//batch_process_size)*batch_process_size != b:
            raise ValueError(f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if node_terminate_thresh is None and split_depth is None:
        raise ValueError("must specify at least one of node_terminate_thresh or split_depth as a terminating condition")
    if node_terminate_thresh is None:
        node_terminate_thresh = 999999999

    d = lower.shape[-1]
    B = batch_process_size

    # print(f"\n == CONSTRUCTING LEVELSET TREE")
    # print(f"  node thresh: {n_node_thresh}")n_node_thresh

    # Initialize data
    node_lower = lower[None,:]
    node_upper = upper[None,:]
    node_valid = jnp.ones((1,), dtype=bool)
    N_curr_nodes = 1
    N_total_computed_cells = 0
    N_skipped = 0
    current_volume_size = 1
    volume_skipped = 0
    finished_interior_lower = jnp.zeros((batch_process_size,3)) if with_interior_nodes else None
    finished_interior_upper = jnp.zeros((batch_process_size,3)) if with_interior_nodes else None
    N_finished_interior = 0
    finished_exterior_lower = jnp.zeros((batch_process_size,3)) if with_exterior_nodes else None
    finished_exterior_upper = jnp.zeros((batch_process_size,3)) if with_exterior_nodes else None
    N_finished_exterior = 0
    N_func_evals = 0

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth+1 # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Reshape in to batches of size <= B
        init_bucket_size = node_lower.shape[0]
        this_b = min(B, init_bucket_size)
        N_func_evals += node_lower.shape[0]
        # utils.printarr(node_valid)
        N_input_node = node_valid.sum()
        N_total_computed_cells += N_input_node
        node_valid = jnp.reshape(node_valid, (-1, this_b))
        node_lower = jnp.reshape(node_lower, (-1, this_b, d))
        node_upper = jnp.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr_nodes / this_b)) # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = (N_curr_nodes >= node_terminate_thresh) or i_split+1 == n_splits
        do_continue_splitting = not quit_next

        # print(f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # enlarge the finished nodes if needed
        if with_interior_nodes:
            while finished_interior_lower.shape[0] - N_finished_interior < N_curr_nodes:
                finished_interior_lower = utils.resize_array_axis(finished_interior_lower, 2*finished_interior_lower.shape[0])
                finished_interior_upper = utils.resize_array_axis(finished_interior_upper, 2*finished_interior_upper.shape[0])
        if with_exterior_nodes:
            while finished_exterior_lower.shape[0] - N_finished_exterior < N_curr_nodes:
                finished_exterior_lower = utils.resize_array_axis(finished_exterior_lower, 2*finished_exterior_lower.shape[0])
                finished_exterior_upper = utils.resize_array_axis(finished_exterior_upper, 2*finished_exterior_upper.shape[0])

        # map over the batches
        out_valid = jnp.zeros((nb, 2*this_b), dtype=bool)
        out_lower = jnp.zeros((nb, 2*this_b, d))
        out_upper = jnp.zeros((nb, 2*this_b, d))
        total_n_valid = 0
        # our prob estimation
        num_grid = 8 ** 3 * 2 ** (n_splits - i_split)
        # num_grid = 8 ** 3 * (n_splits - i_split)
        # num_grid = 2 ** (n_splits - i_split)

        num_grid = min(2 ** 31 - 1, num_grid)

        for ib in range(n_occ): 
            out_valid, out_lower, out_upper, total_n_valid, \
            finished_interior_lower, finished_interior_upper, N_finished_interior, \
            finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
            = \
            construct_uniform_unknown_levelset_tree_iter(func, params, do_continue_splitting, \
                    node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], \
                    ib, out_valid, out_lower, out_upper, total_n_valid, \
                    finished_interior_lower, finished_interior_upper, N_finished_interior, \
                    finished_exterior_lower, finished_exterior_upper, N_finished_exterior, \
                    isovalue=isovalue, offset=offset, num_grid = num_grid, prob_threshold=prob_threshold)

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper
        N_curr_nodes = total_n_valid
        if not quit_next:
            skipped = N_input_node - total_n_valid/2
        else:
            skipped = N_input_node - total_n_valid
        N_skipped += skipped
        volume_skipped += current_volume_size * skipped
        current_volume_size /= 2

        # flatten back out
        node_valid = jnp.reshape(node_valid, (-1,))
        node_lower = jnp.reshape(node_lower, (-1, d))
        node_upper = jnp.reshape(node_upper, (-1, d))

        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid) 
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid, target_bucket_size, node_lower, node_upper)

        if quit_next:
            break


    # pack the output in to a dict to support optional outputs
    out_dict = {
            'unknown_node_valid' : node_valid,
            'unknown_node_lower' : node_lower,
            'unknown_node_upper' : node_upper,
        }

    if with_interior_nodes:
        out_dict['interior_node_valid'] = jnp.arange(finished_interior_lower.shape[0]) < N_finished_interior
        out_dict['interior_node_lower'] = finished_interior_lower
        out_dict['interior_node_upper'] = finished_interior_upper

    if with_exterior_nodes:
        out_dict['exterior_node_valid'] = jnp.arange(finished_exterior_lower.shape[0]) < N_finished_exterior
        out_dict['exterior_node_lower'] = finished_exterior_lower
        out_dict['exterior_node_upper'] = finished_exterior_upper

    return out_dict, N_total_computed_cells, N_skipped, volume_skipped/N_skipped if N_skipped != 0 else 0


if __name__ == "__main__":
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_enable_x64", True)
    data_bound = 1
    isovalue = 0
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound,   data_bound,  data_bound))
    t=5

    data_opts = ['vorts', 'asteroid', 'combustion', 'ethanediol','isotropic','fox', 'hammer','birdcage','bunny']
############################################
    data_type = 3
    n_mc_depth = 8
############################################
    if data_type == 0:
        test_model = 'sample_inputs/vorts_sin_8_32.npz'
        # test_model = 'sample_inputs/vorts_sin_8_64.npz'
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
        test_model = 'sample_inputs/iso_sin_3_128.npz'
        # test_model = 'sample_inputs/iso_sin_5_128.npz'
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

    n_mc_subcell= 3  #larger value may be useful for larger networks
    batch_process_size = 2 ** 12

    modes = ['uncertainty_all', 'affine_ua', 'affine_all', 'affine_fixed', 'affine_truncate', 'affine_append']
    mode = modes[0]
    affine_opts = {}
    affine_opts['affine_n_truncate'] = 8
    affine_opts['affine_n_append'] = 4
    affine_opts['sdf_lipschitz'] = 1.
    truncate_policies = ['absolute', 'relative']
    affine_opts['affine_truncate_policy'] = truncate_policies[0]

    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound, data_bound, data_bound))

    print('[Data]', data_opts[data_type])
    print(f"[Max depth] {n_mc_depth}")
    print(f"[Subcell depth] {n_mc_subcell}")

    for i in range(6):
        mode = modes[i]
        print('[mode]', mode)

        implicit_func, params = generate_implicit_from_file(test_model, mode=mode, **affine_opts)

        side_n_cells = 64
        side_n_pts = (1+side_n_cells)
        side_coords0 = jnp.linspace(-1,1, num=side_n_pts)
        side_coords1 = jnp.linspace(-1,1, num=side_n_pts)
        side_coords2 = jnp.linspace(-1,1, num=side_n_pts)
        grid_coords0, grid_coords1, grid_coords2 = jnp.meshgrid(side_coords0[0:-1], side_coords1[0:-1], side_coords2[0:-1], indexing='ij')
        node_lower = jnp.stack((grid_coords0.flatten(), grid_coords1.flatten(), grid_coords2.flatten()), axis=-1)
        grid_coords0, grid_coords1, grid_coords2 = jnp.meshgrid(side_coords0[1:], side_coords1[1:], side_coords2[1:], indexing='ij')
        node_upper = jnp.stack((grid_coords0.flatten(), grid_coords1.flatten(), grid_coords2.flatten()), axis=-1)
        node_valid = jnp.ones((node_lower.shape[0]),jnp.bool_)

        print(node_lower.shape)

        #warm up
        # out_dict, N_total_computed, N_skipped, average_volume = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=3*(n_mc_depth-n_mc_subcell), \
        #                                                         isovalue=isovalue, batch_process_size=batch_process_size, prob_threshold=t)
        # jax.block_until_ready(out_dict)
        

        def eval_one_node(lower, upper):
            # perform an affine evaluation
            node_type = implicit_func.classify_box(params, lower, upper, isovalue=isovalue, offset=0., prob_threshold=t, num_grid=1)
            return node_type
            
        # evaluate the function inside nodes
        # warm up
        for l, u in zip(node_lower.reshape(-1,4096,3), node_upper.reshape(-1,4096,3)):
            node_types = jax.vmap(eval_one_node)(l, u)
            jax.block_until_ready(node_types)
        # node_types = jax.vmap(eval_one_node)(node_lower, node_upper)

        t1 = time.time()
        for l, u in zip(node_lower.reshape(-1,4096,3), node_upper.reshape(-1,4096,3)):
            node_types = jax.vmap(eval_one_node)(l, u)
            jax.block_until_ready(node_types)
        # node_types = jax.vmap(eval_one_node)(node_lower, node_upper)
        # jax.block_until_ready(node_types)
        query_time = time.time() - t1


        side_n_cells = 128
        side_n_pts = (1+side_n_cells)
        side_coords0 = jnp.linspace(-1,1, num=side_n_pts)
        side_coords1 = jnp.linspace(-1,1, num=side_n_pts)
        side_coords2 = jnp.linspace(-1,1, num=side_n_pts)
        grid_coords0, grid_coords1, grid_coords2 = jnp.meshgrid(side_coords0[0:-1], side_coords1[0:-1], side_coords2[0:-1], indexing='ij')
        node_lower2 = jnp.stack((grid_coords0.flatten(), grid_coords1.flatten(), grid_coords2.flatten()), axis=-1)
        grid_coords0, grid_coords1, grid_coords2 = jnp.meshgrid(side_coords0[1:], side_coords1[1:], side_coords2[1:], indexing='ij')
        node_upper = jnp.stack((grid_coords0.flatten(), grid_coords1.flatten(), grid_coords2.flatten()), axis=-1)
        node_valid = jnp.ones((node_lower.shape[0]),jnp.bool_)
        print(node_lower2.shape)

        # vals = jax.vmap(partial(implicit_func, params))(node_lower2)
        vfunc = jax.vmap(partial(implicit_func, params))
        vals = jax.lax.map(vfunc, node_lower2.reshape(-1, 4096, 3))
        vals.block_until_ready()

        t1 = time.time()
        # vals = jax.vmap(partial(implicit_func, params))(node_lower2)
        vfunc = jax.vmap(partial(implicit_func, params))
        vals = jax.lax.map(vfunc, node_lower2.reshape(-1, 4096, 3))
        vals.block_until_ready()
        dense_time = time.time() - t1

        mul = query_time / dense_time * (node_lower2.shape[0] / node_lower.shape[0])
        print("%.1f" % mul)

        # out_dict, N_total_computed, N_skipped, average_volume = construct_uniform_unknown_levelset_tree(implicit_func, params, lower, upper, split_depth=3*(n_mc_depth-n_mc_subcell), \
        #                                                         isovalue=isovalue, batch_process_size=batch_process_size, prob_threshold=t)
        # print(average_volume)
