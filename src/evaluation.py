import jax
import jax.numpy as jnp

from functools import partial
import math
import numpy as np
import numba as nb

import utils
from bucketing import *
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import extract_cell
from kd_tree import construct_uniform_unknown_levelset_tree

def compare_mc_clt(func, params, lower, upper, n=1e6, batch_process_size=4096):
    # generate point for dense reconstruction
    # with utils.Timer("Dense"):
    n = int(n)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    #uniform
    rand_n = jax.random.uniform(subkey,(n,3))
    delta = upper - lower
    coords = lower[None,:] + delta[None,:] * rand_n
    #normal
    # rand_n = jax.random.normal(subkey,(n,3))
    # center = (upper + lower) / 2
    # delta = upper - lower
    # sigma = jnp.sqrt(delta ** 2 / 12)
    # coords = center + sigma * rand_n
    dense_vals = utils.evaluate_implicit_fun(func, params, coords, batch_process_size)
    # PAF
    # with utils.Timer("PAF"):
    mu, sigma = func.get_paf(params, lower, upper,)
    return dense_vals, mu, sigma

def iso_voxels(np_array,iso_value, get_level = False):
    return_array,level_volume, total_access = iso_voxels_numba(np_array, iso_value, get_level)
    if get_level:
        return return_array,level_volume, total_access
    else:
        return return_array

# function to get the gt iso values
@nb.jit(nopython=True)
def iso_voxels_numba(np_array,iso_value, get_level = False):
    grid_res = np_array.shape[0]
    vol_res = grid_res -1
    neg_array = np_array < iso_value
    pos_array = np_array >= iso_value
    if get_level:
        ideal_grid = np.zeros_like(np_array,dtype=np.int32)
        ideal_volume = np.zeros((vol_res,vol_res,vol_res),dtype=np.int32)
    return_array = []
    for i in range(vol_res):
        for j in range(vol_res):
            for k in range(vol_res):
                local_idx = (
                    (i,i,i,i,i+1,i+1,i+1,i+1),
                    (j,j,j+1,j+1,j,j,j+1,j+1),
                    (k,k+1,k,k+1,k,k+1,k,k+1),
                )
                local_idx = np.array(local_idx).T
                neg = False
                pos = False
                for idx in local_idx:
                    if neg_array[idx[0],idx[1],idx[2]]:
                        neg = True
                    if pos_array[idx[0],idx[1],idx[2]]:
                        pos = True
                if pos and neg:
                    return_array.append(i * vol_res * vol_res + j * vol_res + k)
                    if get_level:
                        ideal_volume[i,j,k] = 1
                        # also mark all grids as true to calculate 
                        for idx in local_idx:
                            ideal_grid[idx[0],idx[1],idx[2]] = 1
    if get_level:
        level_volume, total_access = get_level_acess(ideal_volume, ideal_grid)
    return return_array,level_volume, total_access

@nb.jit(nopython=True)
def get_level_acess(ideal_volume, ideal_grid):
    res = ideal_volume.shape[0]
    new_res = res
    last_volume = ideal_volume
    level_volume = ideal_volume.copy()
    while True:
        new_res = new_res // 2
        if new_res == 0:
            break
        new_volume = np.zeros((new_res,new_res,new_res), np.int32)
        for l in range(new_res):
            for m in range(new_res):
                for n in range(new_res):
                    i = 2 * l
                    j = 2 * m
                    k = 2 * n
                    local_idx = (
                        (i,i,i,i,i+1,i+1,i+1,i+1),
                        (j,j,j+1,j+1,j,j,j+1,j+1),
                        (k,k+1,k,k+1,k,k+1,k,k+1),
                    )
                    local_idx = np.array(local_idx).T
                    for idx in local_idx:
                        if last_volume[idx[0],idx[1],idx[2]]:
                            new_volume[l,m,n] = 1
                            # update new grid
                            for it1 in range(3):
                                for it2 in range(3):
                                    for it3 in range(3):
                                        block_size = res // new_res 
                                        # we mark all children
                                        cx = int(l * block_size + it1 * (block_size // 2))
                                        cy = int(m * block_size + it2 * (block_size // 2))
                                        cz = int(n * block_size + it3 * (block_size // 2))
                                        ideal_grid[cx, cy, cz] = 1
                            #update level volume
                            level_volume[
                                l*block_size:(l+1)*block_size,
                                m*block_size:(m+1)*block_size,
                                n*block_size:(n+1)*block_size,
                                ] += 1
                            break
        last_volume = new_volume

        # volume_list.append(new_volume)
        # grid_list.append(new_grid)
    total_access = ideal_grid.sum()
    return level_volume, total_access

def get_ci_mc(aff_matrix, prob=0.95, mc_number = 1000): #monte carlo sampling
    key = jax.random.PRNGKey(42)
    samples = jax.random.uniform(key, (mc_number, *aff_matrix.shape)) # (mc_number, batch_size, n_aff)
    samples = samples * 2 - 1
    radius = (samples * aff_matrix[None,...]).sum(-1)  # (mc_number, batch_size)
    radius = jnp.sort(radius, axis=0)
    idx = int((1-prob) / 2 * mc_number)
    return radius[idx], radius[-1-idx]


def hierarchical_iso_voxels(func, params, isovalue, lower, upper, depth, n_subcell_depth=2, batch_process_size = 2 ** 20, t = 1.):

    # Build a tree over the isosurface
    # By definition returned nodes are all SIGN_UNKNOWN, and all the same size
    out_dict = construct_uniform_unknown_levelset_tree(func, params, lower, upper, \
        split_depth=3*(depth-n_subcell_depth), isovalue=isovalue, batch_process_size=batch_process_size, prob_threshold=t)
    node_valid = out_dict['unknown_node_valid']
    node_lower = out_dict['unknown_node_lower']
    node_upper = out_dict['unknown_node_upper']

    mask = np.zeros((2**depth,2**depth,2**depth), dtype=np.bool_)
    delta = (upper - lower) / (2 ** depth)
    start_idx = jnp.round((node_lower[node_valid]-lower) / delta).astype(jnp.int32)
    start_idx = np.asarray(start_idx)
    end_idx = jnp.round((node_upper[node_valid]-lower) / delta).astype(jnp.int32)
    end_idx = np.asarray(end_idx)
    mask = assign_helper(mask, start_idx, end_idx)
    # length = 2** n_subcell_depth
    # mask[tuple(start_idx.T.tolist())] = True
    # mask = mask.repeat(length,0)
    # mask = mask.repeat(length,1)
    # mask = mask.repeat(length,2)
    return mask

@nb.jit(nopython=True)
def assign_helper(mask, start_idx, end_idx):
    for s, e in zip(start_idx, end_idx):
        mask[s[0]:e[0],s[1]:e[1],s[2]:e[2]] = True
    return mask
   

@nb.jit(nopython=True)
def get_real_bounds_helper(values, all_node_lower, all_node_upper, isovalue = 0, coord_bound = 1):
    node_type = np.zeros((all_node_lower.shape[0],), np.int32)
    worst_dim = np.argmax(all_node_upper - all_node_lower, axis = -1)
    # ignore the valid mask for the output
    
    for i in range(len(all_node_lower)):
        # print("%d/%d" % (i,len(all_node_lower)), end='\r')
        lower = all_node_lower[i]
        upper = all_node_upper[i]

        max_coord = values.shape[0] + 1
        lower_coord = ((lower + coord_bound) * max_coord/ 2).astype(np.int32)
        l1,l2,l3 = lower_coord
        upper_coord = ((upper + coord_bound) * max_coord/ 2).astype(np.int32)
        u1,u2,u3 = upper_coord
        vs = values[l1:u1+1, l2:u2+1, l3:u3+1]

        if vs.max() < isovalue:
            node_type[i] = SIGN_NEGATIVE
        elif vs.min() > isovalue:
            node_type[i] = SIGN_POSITIVE
        else:
            node_type[i] = SIGN_UNKNOWN
        
    return node_type, worst_dim

@nb.jit(nopython=True)
def generate_oct_min_max(data, subcell_level = 3):
    subcell_side_n = 2 ** subcell_level
    n, _, _ = data.shape  # n should be 2^k + 1
    level = int(np.log2(n-1))
    
    return level

def kd_tree_array_iter_dense(
        vals, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper,
        isovalue=0.,
        ):

    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]
        
    # evaluate the function inside nodes
    node_types, node_split_dim = get_real_bounds_helper(vals, np.asarray(node_lower), np.asarray(node_upper), isovalue)
    

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])

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
        outL = out_valid.shape[1]

    else:
        node_valid = jnp.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid = out_valid.at[ib,:outL].set(node_valid)
    out_lower = out_lower.at[ib,:outL,:].set(node_lower)
    out_upper = out_upper.at[ib,:outL,:].set(node_upper)

    return out_valid, out_lower, out_upper

@partial(jax.jit, static_argnames=("func","continue_splitting"), donate_argnums=(7,8,9))
def kd_tree_array_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper,
        isovalue=0.,offset=0.,prob_threshold=0.95,num_grid=1, 
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
    

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = utils.logical_and_all([node_valid, node_types == SIGN_UNKNOWN])

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
        outL = out_valid.shape[1]

    else:
        node_valid = jnp.logical_and(node_valid, node_types == SIGN_UNKNOWN)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid = out_valid.at[ib,:outL].set(node_valid)
    out_lower = out_lower.at[ib,:outL,:].set(node_lower)
    out_upper = out_upper.at[ib,:outL,:].set(node_upper)

    return out_valid, out_lower, out_upper


def kd_tree_array(func, params, split_depth=None, isovalue=0., offset=0., batch_process_size=4096, prob_threshold = 1., dense = False, vals = None):
       
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b//batch_process_size)*batch_process_size != b:
            raise ValueError(f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    if split_depth is None:
        raise ValueError("must specify split_depth as a terminating condition")

    d = 3
    B = batch_process_size
    split_order = (0,1,2)
    kd_array = jnp.zeros((2 ** (split_depth+1) - 1,), dtype = bool)
    if dense and vals is None:
        raise ValueError("must specify values in the dense mode")


    ## Recursively build the tree
    i_split = 0
    start_index = 0
    node_valid = jnp.ones((1,), bool)
    node_lower = jnp.asarray([-1,-1,-1])[None,:]
    node_upper = jnp.asarray([1,1,1])[None,:]
    n_splits = split_depth+1 # 1 extra because last round doesn't split
    for i_split in range(n_splits):

        # Reshape in to batches of size <= B
        level_size = 2 ** i_split
        # input_array = kd_array[start_index: start_index + level_size]
        end_index = start_index + level_size


        # utils.printarr(node_valid, node_lower, node_upper)
        N_curr_nodes = node_valid.shape[0]
        this_b = min(B, level_size)
        node_valid = jnp.reshape(node_valid, (-1, this_b))
        node_lower = jnp.reshape(node_lower, (-1, this_b, d))
        node_upper = jnp.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr_nodes / this_b)) # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = i_split+1 == n_splits
        do_continue_splitting = not quit_next

        # print(f"Kd-tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {level_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # map over the batches
        out_valid = jnp.zeros((nb, 2*this_b), dtype=bool)
        out_lower = jnp.zeros((nb, 2*this_b, 3))
        out_upper = jnp.zeros((nb, 2*this_b, 3))

        # our prob estimation
        num_grid = 8 ** 3 * 2 ** (n_splits - i_split)

        num_grid = min(2 ** 31 - 1, num_grid)

        for ib in range(n_occ): 
            if dense:
                out_valid, out_lower, out_upper, \
                = \
                kd_tree_array_iter_dense(vals, do_continue_splitting, \
                        node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], \
                        ib, out_valid, out_lower, out_upper, \
                        isovalue=isovalue)
            else:
                out_valid, out_lower, out_upper, \
                = \
                kd_tree_array_iter(func, params, do_continue_splitting, \
                        node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], \
                        ib, out_valid, out_lower, out_upper, \
                        isovalue=isovalue, offset=offset, num_grid = num_grid, prob_threshold=prob_threshold)

        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper

        # flatten back out
        split_mask = jnp.reshape(node_valid[:, :this_b], (-1,))
        kd_array = kd_array.at[start_index:end_index].set(split_mask)
        start_index = end_index
        node_valid = jnp.reshape(node_valid, (-1,))
        node_lower = jnp.reshape(node_lower, (-1, d))
        node_upper = jnp.reshape(node_upper, (-1, d))


        if quit_next:
            break


    return kd_array
