import igl

import sys
from functools import partial
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

# Imports from this project
from utils import *
import mlp
import geometry
import render
import queries
import affine

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

def main():

    parser = argparse.ArgumentParser()

    # Build arguments
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
   
    # network
    parser.add_argument("--activation", type=str, default='elu')
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--layer_width", type=int, default=32)
    parser.add_argument("--positional_encoding", action='store_true')
    parser.add_argument("--positional_count", type=int, default=10)
    parser.add_argument("--positional_pow_start", type=int, default=-3)

    # loss / data
    parser.add_argument("--n_epochs", type=int, default=100)
    # parser.add_argument("--n_samples", type=int, default=128 * 128 * 128)
    parser.add_argument("--data_type", type=str, default='vorts', choices=['vorts', 'asteroid', 'combustion', 'ethanediol', 'isotropic'])
    
    # training
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr_decay_every", type=int, default=99999)
    parser.add_argument("--lr_decay_frac", type=float, default=.5)

    # jax options
    parser.add_argument("--log-compiles", action='store_true')
    parser.add_argument("--disable-jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable-double-precision", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # validate some inputs
    if args.activation not in ['relu', 'elu', 'sin']:
        raise ValueError("unrecognized activation")
    if not args.output_file.endswith('.npz'):
        raise ValueError("output file should end with .npz")

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())
   
    # Set jax things
    if args.log_compiles:
        jax.config.update("jax_log_compiles", 1)
    if args.disable_jit:
        jax.config.update('jax_disable_jit', True)
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.enable_double_precision:
        jax.config.update("jax_enable_x64", True)
   
    # load the input
    print(f"Loading data {args.input_file}")
    data = load_data(args.data_type, args.input_file)

    # preprocess (normalized to -1,1)
    # data = jnp.log2(data+1)
    std, mean = data.std(), data.mean()
    data = (data - mean) / std
    data_max, data_min = data.max(), data.min()
    # data = (data - data_min) / (data_max - data_min)
    # data = (data - 0.5) * 2
    print(data.shape, data.min(), data.max())
    print(f"  ...done")

    # sample training points
    print(f"Sampling {len(data.flatten())} training points...")
    samp, samp_v = sample_volume(data.shape, data)

    samp_target = samp_v
    samp_weight = jnp.ones_like(samp_target)

    print(f"  ...done")

    # construct the network 
    print(f"Constructing {args.n_layers}x{args.layer_width} {args.activation} network...")
    if args.positional_encoding:
        spec_list = [mlp.pow2_frequency_encode(args.positional_count, start_pow=args.positional_pow_start, with_shift=True), mlp.sin()]
        layers = [6*args.positional_count] + [args.layer_width]*args.n_layers + [1]
        spec_list += mlp.quick_mlp_spec(layers, args.activation)
    else:
        layers = [3] + [args.layer_width]*args.n_layers + [1]
        spec_list = mlp.quick_mlp_spec(layers, args.activation)
    orig_params = mlp.build_spec(spec_list) 
    implicit_func = mlp.func_from_spec()


    # layer initialization
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    orig_params = mlp.initialize_params(orig_params, subkey)
    print(f"  ...done")

    # test eval to ensure the function isn't broken
    print(f"Network test evaluation...")
    implicit_func(orig_params, jnp.array((0.1, 0.2, 0.3)))
    print(f"...done")

    # Create an optimizer
    print(f"Creating optimizer...")
    def step_func(i_epoch): 
        out = args.lr * (args.lr_decay_frac ** (i_epoch // args.lr_decay_every))
        return out
    opt = optimizers.adam(step_func)

    opt_param_keys = mlp.opt_param_keys(orig_params)

    # Union our optimizable parameters with the non-optimizable ones
    def add_full_params(opt_params):
        all_params = opt_params
        
        for k in orig_params:
            if k not in all_params:
                all_params[k] = orig_params[k]
    
    # Union our optimizable parameters with the non-optimizable ones
    def filter_to_opt_params_only(all_params):
        for k in all_params:
            if k not in opt_param_keys:
                del all_params[k]
    
    # Construct the optimizer over the optimizable params
    opt_params_only = {}
    for k in mlp.opt_param_keys(orig_params):
        opt_params_only[k] = orig_params[k]
    opt_state = opt.init_fn(opt_params_only)
    print(f"...done")

    best_loss = float('inf')
    best_params = None



    @jax.jit
    def generate_batch(rngkey, samples_in, samples_out, samples_weight):

        # concatenate to make processing easier
        # samples = jnp.concatenate((samples_in, samples_out[:,None], samples_weight[:,None]), axis=-1)

        # shuffle
        # samples = jax.random.permutation(rngkey, samples, axis=0)
        shuffled_index = jax.random.permutation(rngkey, samples_in.shape[0])

        # split in to batches
        # (discard any extra samples)
        batch_count = samples_in.shape[0] // args.batch_size
        n_batch_total = args.batch_size * batch_count
        # samples = samples[:n_batch_total, :]
        shuffled_index = shuffled_index[:n_batch_total]

        # split back up
        # samples_in = samples[:,:3]
        # samples_out = samples[:,3]
        # samples_weight = samples[:,4]
        samples_in = samples_in[shuffled_index]
        samples_out = samples_out[shuffled_index]
        samples_weight = samples_weight[shuffled_index]

        batch_in = jnp.reshape(samples_in, (batch_count, args.batch_size, 3))
        batch_out = jnp.reshape(samples_out, (batch_count, args.batch_size))
        batch_weight = jnp.reshape(samples_weight, (batch_count, args.batch_size))

        return batch_in, batch_out, batch_weight, batch_count
    
    def batch_loss_fn(params, batch_coords, batch_target, batch_weight):

        add_full_params(params)
   
        def loss_one(params, coords, target, weight):
            pred = implicit_func(params, coords)
            return (pred - target) ** 2 * weight
        
        loss_terms = jax.vmap(partial(loss_one, params))(batch_coords, batch_target, batch_weight)
        loss_mean = jnp.mean(loss_terms)
        return loss_mean

    def cal_PSNR(params, batches_in, batches_out):

        add_full_params(params)
        total_sse = 0
        n_total = 0
   
        def sse(params, coords, targets):
            pred = implicit_func(params, coords)
            return (pred - targets) ** 2
        
        for i in range(len(batches_in)):
            coords, targets = batches_in[i], batches_out[i]
            sse_terms = jax.vmap(partial(sse, params))(coords, targets)
            total_sse += float(jnp.sum(sse_terms))
            n_total += len(coords)
        mse = total_sse / n_total
        psnr = 10 * jnp.log10((data_max - data_min)**2/mse)
        return psnr

    @jax.jit
    def train_step(i_epoch, i_step, opt_state, batch_in, batch_out, batch_weight):
   
        opt_params = opt.params_fn(opt_state)
        value, grads = jax.value_and_grad(batch_loss_fn)(opt_params, batch_in, batch_out, batch_weight)
        opt_state = opt.update_fn(i_epoch, grads, opt_state)
        
        return value, opt_state
    
    #try creating the batch before
    key, subkey = jax.random.split(key)
    batches_in, batches_out, batches_weight, n_batches = generate_batch(subkey, samp, samp_target, samp_weight)

    print(f"Training...")
    i_step = 0
    for i_epoch in range(args.n_epochs):
        losses = []
        n_total = 0

        for i_b in range(n_batches):
            actual_batch_size = len(batches_in[i_b,...])

            loss, opt_state = train_step(i_epoch, i_step, opt_state, batches_in[i_b,...], batches_out[i_b,...], batches_weight[i_b,...])

            loss = float(loss)
            losses.append(loss * actual_batch_size) # add up the batch size
            n_total += actual_batch_size
            i_step += 1
        mean_loss = np.sum(np.array(losses)) / n_total
        # directly use the average loss to calculate
        psnr = 10 * jnp.log10((data_max - data_min)**2/mean_loss)
        # opt_params = opt.params_fn(opt_state)
        # psnr = cal_PSNR(opt_params, batches_in, batches_out) 

        print(f"== Epoch {i_epoch} / {args.n_epochs}   loss: {mean_loss:.6f}  PSNR: {psnr}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = opt.params_fn(opt_state)
            add_full_params(best_params)
            print("  --> new best")

            print(f"Saving result to {args.output_file}")
            mlp.save(args.output_file, best_params)
            print(f"  ...done", flush=True)

    
    # save the result
    print(f"Saving result to {args.output_file}")
    mlp.save(args.output_file, best_params)
    print(f"  ...done")


if __name__ == '__main__':
    main()
