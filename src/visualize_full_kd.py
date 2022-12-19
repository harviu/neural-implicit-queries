import jax
import jax.numpy as jnp

from functools import partial
import math

import numpy as np

import utils
from bucketing import *
import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE
import extract_cell
import geometry
import implicit_mlp_utils

@partial(jax.jit, static_argnames=("func","continue_splitting"), donate_argnums=(7,8,9,10,11,12))
def construct_full_kd_tree_iter(
        func, params, continue_splitting,
        node_valid, node_lower, node_upper,
        ib, out_valid, out_lower, out_upper, out_n_valid, out_lb, out_ub,
        ):

    N_in = node_lower.shape[0]
    d = node_lower.shape[-1]

    def eval_one_node(lower, upper):

        # perform an affine evaluation
        lb, ub = func.estimate_box_bounds(params, lower, upper)

        # use the largest length along any dimension as the split policy
        worst_dim = jnp.argmax(upper-lower, axis=-1)

        return lb, ub, worst_dim
        
    # evaluate the function inside nodes
    lb, ub, node_split_dim = jax.vmap(eval_one_node)(node_lower, node_upper)

    # split the unknown nodes to children
    # (if split_children is False this will just not create any children at all)
    split_mask = node_valid
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
        new_N_valid = jnp.sum(node_valid)
        outL = node_valid.shape[0]

    # write the result in to arrays
    # utils.printarr(out_valid, node_valid, out_lower, node_lower, out_upper, node_upper)
    out_valid = out_valid.at[ib,:outL].set(node_valid)
    out_lower = out_lower.at[ib,:outL,:].set(node_lower)
    out_upper = out_upper.at[ib,:outL,:].set(node_upper)
    out_lb = out_lb.at[ib,:outL].set(lb)
    out_ub = out_ub.at[ib,:outL].set(ub)
    out_n_valid = out_n_valid + new_N_valid

    return out_valid, out_lower, out_upper, out_n_valid, out_lb, out_ub


def construct_full_kd_tree(func, params, lower, upper, split_depth=12, batch_process_size=2048):
       
    # Validate input
    # ASSUMPTION: all of our bucket sizes larger than batch_process_size must be divisible by batch_process_size
    for b in bucket_sizes:
        if b > batch_process_size and (b//batch_process_size)*batch_process_size != b:
            raise ValueError(f"batch_process_size must be a factor of our bucket sizes, is not a factor of {b} (try a power of 2)")
    node_terminate_thresh = 999999999

    d = lower.shape[-1]
    B = batch_process_size

    print(f"\n == CONSTRUCTING LEVELSET TREE")
    # print(f"  node thresh: {n_node_thresh}")n_node_thresh

    # Initialize data
    node_lower = lower[None,:]
    node_upper = upper[None,:]
    node_valid = jnp.ones((1,), dtype=bool)
    N_curr_nodes = 1
    N_func_evals = 0
    all_nodes_lower = node_lower.copy()
    all_nodes_upper = node_upper.copy()
    all_nodes_valid = node_valid.copy()
    N_total_nodes = 1
    all_lb = jnp.zeros((0,))
    all_ub = jnp.zeros((0,))

    ## Recursively build the tree
    i_split = 0
    n_splits = 99999999 if split_depth is None else split_depth+1 # 1 extra because last round doesn't split
    for i_split in range(n_splits):
        # Reshape in to batches of size <= B
        init_bucket_size = node_lower.shape[0]
        N_curr_nodes = jnp.sum(node_valid)
        this_b = min(B, init_bucket_size)
        N_func_evals += node_lower.shape[0]
        # utils.printarr(node_valid, node_lower, node_upper)
        node_valid = jnp.reshape(node_valid, (-1, this_b))
        node_lower = jnp.reshape(node_lower, (-1, this_b, d))
        node_upper = jnp.reshape(node_upper, (-1, this_b, d))
        nb = node_lower.shape[0]
        n_occ = int(math.ceil(N_curr_nodes / this_b)) # only the batches which are occupied (since valid nodes are densely packed at the start)

        # Detect when to quit. On the last iteration we need to not do any more splitting, but still process existing nodes one last time
        quit_next = (N_curr_nodes >= node_terminate_thresh) or i_split+1 == n_splits
        do_continue_splitting = not quit_next

        print(f"Uniform levelset tree. iter: {i_split}  N_curr_nodes: {N_curr_nodes}  bucket size: {init_bucket_size}  batch size: {this_b}  number of batches: {nb}  quit next: {quit_next}  do_continue_splitting: {do_continue_splitting}")

        # map over the batches
        out_valid = jnp.zeros((nb, 2*this_b), dtype=bool)
        out_lower = jnp.zeros((nb, 2*this_b, 3))
        out_upper = jnp.zeros((nb, 2*this_b, 3))
        out_lb = jnp.zeros((nb, this_b))
        out_ub = jnp.zeros((nb, this_b))
        total_n_valid = 0
        for ib in range(n_occ): 
            out_valid, out_lower, out_upper, total_n_valid, out_lb, out_ub \
            = \
            construct_full_kd_tree_iter(func, params, do_continue_splitting, \
                    node_valid[ib,...], node_lower[ib,...], node_upper[ib,...], \
                    ib, out_valid, out_lower, out_upper, total_n_valid, out_lb, out_ub)


        node_valid = out_valid
        node_lower = out_lower
        node_upper = out_upper

        # flatten back out
        node_valid = jnp.reshape(node_valid, (-1,))
        node_lower = jnp.reshape(node_lower, (-1, d))
        node_upper = jnp.reshape(node_upper, (-1, d))
        out_lb = jnp.reshape(out_lb, (-1,))
        out_ub = jnp.reshape(out_ub, (-1,))        
        out_ub = out_ub[:N_curr_nodes] # Number of nodes in the last ite
        out_lb = out_lb[:N_curr_nodes]

        N_curr_nodes = total_n_valid # update the number


        # Compactify and rebucket arrays
        target_bucket_size = get_next_bucket_size(total_n_valid) 
        node_valid, N_curr_nodes, node_lower, node_upper = compactify_and_rebucket_arrays(node_valid, target_bucket_size, node_lower, node_upper)

        # concatenate the bounds
        all_lb = jnp.concatenate((all_lb, out_lb))
        all_ub = jnp.concatenate((all_ub, out_ub))

        if quit_next:
            break
        else:
            # concatenate if not quitting
            N_total_nodes += total_n_valid
            all_nodes_lower = jnp.concatenate((all_nodes_lower, node_lower[:N_curr_nodes]))
            all_nodes_upper = jnp.concatenate((all_nodes_upper, node_upper[:N_curr_nodes]))
            all_nodes_valid = jnp.concatenate((all_nodes_valid, node_valid[:N_curr_nodes]))



    # pack the output in to a dict to support optional outputs
    out_dict = {
            'all_node_valid' : all_nodes_valid,
            'all_node_lower' : all_nodes_lower,
            'all_node_upper' : all_nodes_upper,
            'lb': all_lb,
            'ub': all_ub,
        }

    return out_dict

def get_real_bounds_helper(values, lower, upper, coord_bound = 1):
    max_coord = values.shape[0] + 1
    lower_coord = ((lower + coord_bound) * max_coord/ 2).astype(int)
    l1,l2,l3 = lower_coord
    upper_coord = ((upper + coord_bound) * max_coord/ 2).astype(int)
    u1,u2,u3 = upper_coord
    vs = values[l1:u1+1, l2:u2+1, l3:u3+1]
    return vs.min(), vs.max()


def get_real_bounds(func, params, output, split_depth=12, subcell_depth=3, batch_process_size=2048):
    d = output['all_node_lower'].shape[-1]
    kd_depth = split_depth // d
    total_depth = kd_depth + subcell_depth
    
    grid_res = (2 ** total_depth) + 1
    ax_coords = jnp.linspace(-1., 1., grid_res)
    grid_x, grid_y, grid_z = jnp.meshgrid(ax_coords, ax_coords, ax_coords, indexing='ij')
    grid = jnp.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=-1)
    sdf_vals = utils.evaluate_implicit_fun(func, params, grid, batch_process_size)
    sdf_vals = sdf_vals.reshape(grid_res, grid_res, grid_res)
    output['real_ub'] = []
    output['real_lb'] = []

    # ignore the valid mask for the output
    for i in range(len(output['all_node_valid'])):
        print("%d/%d" % (i,len(output['all_node_valid'])), end='\r')
        lower = output['all_node_lower'][i]
        upper = output['all_node_upper'][i]
        minimum, maximum = get_real_bounds_helper(sdf_vals, lower, upper)
        output['real_ub'].append(maximum)
        output['real_lb'].append(minimum)
    output['real_ub'] = jnp.asarray(output['real_ub'], dtype=jnp.float32)
    output['real_lb'] = jnp.asarray(output['real_lb'], dtype=jnp.float32)
    return output

def generate_diff_tree(output, split_depth=12, data_bound = 1):
    d = output['all_node_lower'].shape[-1]
    kd_depth = split_depth // d
    max_delta = data_bound * 2
    out = [np.zeros((2 ** i,) * 3, dtype= np.float32) for i in range(kd_depth+1)]
    for i in range(len(output['all_node_valid'])):
        # print("%d/%d" % (i,len(output['all_node_valid'])), end='\r')
        lower = output['all_node_lower'][i]
        upper = output['all_node_upper'][i]
        coord_diff = upper - lower
        if coord_diff[0] == coord_diff[1] and coord_diff[1] == coord_diff[2]:
            size = coord_diff[0]
            res = int(round(max_delta / size))
            level = int(math.log2(res))
            c1 = int(round((lower[0] + data_bound) / size))
            c2 = int(round((lower[1] + data_bound) / size))
            c3 = int(round((lower[2] + data_bound) / size))
            out[level][c1,c2,c3] = abs(output['real_ub'][i] - output['ub'][i]) + abs(output['real_lb'][i] - output['lb'][i])
    return out
            



if __name__ == "__main__":
    # data_bound = float(1)
    # lower = jnp.array((-data_bound, -data_bound, -data_bound))
    # upper = jnp.array((data_bound,   data_bound,  data_bound))

    # implicit_func, params = implicit_mlp_utils.generate_implicit_from_file('sample_inputs/vorts.npz', mode='affine_all')
    # output = construct_full_kd_tree(implicit_func, params, lower, upper)
    # print(len(output['ub']), len(output['all_node_valid']), output['ub'].min())
    # jnp.save('output', output, allow_pickle=True)

    # output = get_real_bounds(implicit_func, params, output)
    # jnp.save('output', output, allow_pickle=True)
    output = jnp.load('output.npy', allow_pickle = True).item()
    out = generate_diff_tree(output)
    print(out)
    for i, o in enumerate(out):
        o.tofile("%d.bin" % i)