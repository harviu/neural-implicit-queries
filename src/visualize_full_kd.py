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

if __name__ == "__main__":
    data_bound = float(1)
    lower = jnp.array((-data_bound, -data_bound, -data_bound))
    upper = jnp.array((data_bound,   data_bound,  data_bound))

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file('sample_inputs/vorts.npz', mode='affine_all')
    output = construct_full_kd_tree(implicit_func, params, lower, upper)
    print(len(output['ub']), len(output['all_node_valid']), output['ub'].min())