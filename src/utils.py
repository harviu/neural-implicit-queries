import time
import os
import inspect

from functools import partial

import jax
import jax.numpy as jnp
import datetime

import numpy as np
import numba as nb

import polyscope as ps
import polyscope.imgui as psim
from vtkmodules import all as vtk
from vtkmodules.util import numpy_support


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

def save_vtk(res, delta, data, filename):
    vtk_data = vtk.vtkImageData()
    vtk_data.SetSpacing(delta)
    vtk_data.SetOrigin(0,0,0)
    # vtk_data.SetExtent((0, res[0]-1, 0, res[1]-1, 0, res[2]-1))
    vtk_data.SetDimensions(res)
    vtk_array = numpy_support.numpy_to_vtk(data.flatten())
    vtk_array.SetName('scalar')
    vtk_data.GetPointData().AddArray(vtk_array)
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_data)
    writer.Write()

def get_ci_stack(aff, prob=0.95): # this is not correct
    # aff = np.array(aff) # convert aff to np array
    aff = np.abs(aff) # take the absolute value and make it symmetric
    aff = np.sort(aff)[::-1]
    acc = 0
    acc_prob = 0
    last_acc_prob = 0
    for i, a in enumerate(aff[:-1]):
        h = 1 / a
        next_a = aff[i+1]
        l = a - next_a
        acc += h
        acc_prob += acc * l
        if acc_prob > 1 - prob:
            residual = 1 - prob - last_acc_prob
            residual_length = residual / acc
            out_length = a - residual_length
            break
        last_acc_prob = acc_prob
    return out_length

def get_ci_mc(aff_matrix, prob=0.95, mc_number = 1000): #monte carlo sampling
    key = jax.random.PRNGKey(42)
    samples = jax.random.uniform(key, (mc_number, *aff_matrix.shape)) # (mc_number, batch_size, n_aff)
    samples = samples * 2 - 1
    radius = (samples * aff_matrix[None,...]).sum(-1)  # (mc_number, batch_size)
    radius = jnp.sort(radius, axis=0)
    idx = int((1-prob) / 2 * mc_number)
    return radius[idx], radius[-1-idx]

def load_data(data_type, data_path):
    
    if data_type == 'vorts':
        data = load_vorts_data(128, data_path)
    elif data_type == 'asteroid':
        data = load_asteroid_data(500, data_path)
    elif data_type == 'combustion':
        data = load_combustion_data(data_path)
    elif data_type == 'ethanediol':
        data = load_ethanediol_data(data_path)
    else:
        raise ValueError("not implemented data type")
    return data

def load_vorts_data(res, data_path):
    data = np.fromfile(data_path, '<f4')[3:].reshape(res,res,res)
    return jnp.asarray(data)

def load_asteroid_data(res, data_path):
    data = np.fromfile(data_path, '<f4').reshape(res,res,res)
    return jnp.asarray(data)

def load_combustion_data(data_path):
    # paraview is in zyx order...
    data = np.fromfile(data_path, '<f4').reshape(120, 720, 480)
    # crop the center 330 ^ 3
    # container = np.zeros((330,) * 3, np.float32)
    # container[165 - 60:165 + 60,:,:] = data[:, 360-165:360+165, 240-165:240+165]
    # crop the center
    container = data[:, 360-170:360+170, :]
    return jnp.asarray(container)

def load_ethanediol_data(data_path):
    data = np.fromfile(data_path, '<f4').reshape(134,116,115)
    return jnp.asarray(data)

def build_grid_samples (res):
    if isinstance(res, int):
        X, Y, Z = np.mgrid[0:res, 0:res, 0:res]
    else:
        assert len(res) == 3
        X, Y, Z = np.mgrid[0:res[0], 0:res[1], 0:res[2]]
    full = np.stack([X.flatten(),Y.flatten(),Z.flatten()],axis=-1)
    return full

def normalize_grid_samples(samples, res, keep_ratio = True):
    res = np.array(res)
    if keep_ratio:
        return (samples * 2 - (res - 1)) / (res.max()-1)
    else:
        return (samples * 2 - (res - 1)) / (res - 1)


@partial(jax.jit, static_argnames=("func", "batch_eval_size"))
def evaluate_implicit_fun(func, params, flat_coords, batch_eval_size = 2 ** 20):
    if flat_coords.shape[0] > batch_eval_size:
        # for very large sets, break into batches
        nb = flat_coords.shape[0] // batch_eval_size
        stragglers = flat_coords[nb*batch_eval_size:,:]
        batched_flat_coords = jnp.reshape(flat_coords[:nb*batch_eval_size,:], (-1, batch_eval_size, 3))
        vfunc = jax.vmap(partial(func, params))
        batched_vals = jax.lax.map(vfunc, batched_flat_coords)
        batched_vals = jnp.reshape(batched_vals, (-1,))
        
        # evaluate any stragglers in the very last batch
        straggler_vals = jax.vmap(partial(func,params))(stragglers)

        flat_vals = jnp.concatenate((batched_vals, straggler_vals))
    else:
        flat_vals = jax.vmap(partial(func,params))(flat_coords)
    return flat_vals

def sample_volume(res, data):
    samp = build_grid_samples(res)
    samp_v = data[tuple(samp.T)]
    samp = normalize_grid_samples(samp, res, keep_ratio = False)
    samp = jnp.asarray(samp, dtype= jnp.float32)
    samp_v = jnp.asarray(samp_v, dtype=jnp.float32)
    return samp, samp_v

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name 
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.3f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename, 'a') as file:
                print(str(datetime.datetime.now()) + ": ", message, file=file)

# Extends dict{} to allow access via dot member like d.elem instead of d['elem']
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


# === Polyscope

def combo_string_picker(name, curr_val, vals_list):

    changed = psim.BeginCombo(name, curr_val)
    clicked = False
    if changed:
        for val in vals_list:
            _, selected = psim.Selectable(val, curr_val==val)
            if selected:
                curr_val=val
                clicked = True
        psim.EndCombo()

    return clicked, curr_val

# === JAX helpers

# quick printing
def printarr(*arrs, data=True, short=True, max_width=200):

    # helper for below
    def compress_str(s):
        return s.replace('\n', '')
    name_align = ">" if short else "<"

    # get the name of the tensor as a string
    frame = inspect.currentframe().f_back
    try:
        # first compute some length stats
        name_len = -1
        dtype_len = -1
        shape_len = -1
        default_name = "[unnamed]"
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            name_len = max(name_len, len(name))
            dtype_len = max(dtype_len, len(str(a.dtype)))
            shape_len = max(shape_len, len(str(a.shape)))
        len_left = max_width - name_len - dtype_len - shape_len - 5

        # now print the acual arrays
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            print(f"{name:{name_align}{name_len}} {str(a.dtype):<{dtype_len}} {str(a.shape):>{shape_len}}", end='') 
            if data:
                # print the contents of the array
                print(": ", end='')
                flat_str = compress_str(str(a))
                if len(flat_str) < len_left:
                    # short arrays are easy to print
                    print(flat_str)
                else:
                    # long arrays
                    if short:
                        # print a shortented version that fits on one line
                        if len(flat_str) > len_left - 4:
                            flat_str = flat_str[:(len_left-4)] + " ..."
                        print(flat_str)
                    else:
                        # print the full array on a new line
                        print("")
                        print(a)
            else:
                print("") # newline
    finally:
        del frame



def logical_and_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = jnp.logical_and(out, vals[i])
    return out

def logical_or_all(vals):
    out = vals[0]
    for i in range(1,len(vals)):
        out = jnp.logical_or(out, vals[i])
    return out

def minimum_all(vals):
    '''
    Take elementwise minimum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.min(combined, axis=0)

def maximum_all(vals):
    '''
    Take elementwise maximum of a list of arrays
    '''
    combined = jnp.stack(vals, axis=0)
    return jnp.max(combined, axis=0)

def all_same_sign(vals):
    '''
    Test if all values in an array have (strictly) the same sign
    '''
    return jnp.logical_or(jnp.all(vals < 0), jnp.all(vals > 0))

# Given a 1d array mask, enumerate the nonero entries 
# example:
# in:  [0 1 1 0 1 0]
# out: [X 0 1 X 2 X]
# where X = fill_val
# if fill_val is None, the array lenght + 1 is used
def enumerate_mask(mask, fill_value=None):
    if fill_value is None:
        fill_value = mask.shape[-1]+1
    out = jnp.cumsum(mask, axis=-1)-1
    out = jnp.where(mask, out, fill_value)
    return out


# Returns the first index past the last True value in a mask
def empty_start_ind(mask):
    return jnp.max(jnp.arange(mask.shape[-1]) * mask)+1
    
# Given a list of arrays all of the same shape, interleaves
# them along the first dimension and returns an array such that
# out.shape[0] = len(arrs) * arrs[0].shape[0]
def interleave_arrays(arrs):
    s = list(arrs[0].shape)
    s[0] *= len(arrs)
    return jnp.stack(arrs, axis=1).reshape(s)

@partial(jax.jit, static_argnames=("new_size","axis"))
def resize_array_axis(A, new_size, axis=0):
    first_N = min(new_size, A.shape[0])
    shape = list(A.shape)
    shape[axis] = new_size
    new_A = jnp.zeros(shape, dtype=A.dtype)
    new_A = new_A.at[:first_N,...].set(A.at[:first_N,...].get())
    return new_A

def smoothstep(x):
    out = 3.*x*x - 2.*x*x*x
    out = jnp.where(x < 0, 0., out)
    out = jnp.where(x > 1, 1., out)
    return out

def binary_cross_entropy_loss(logit_in, target):
    # same as the pytorch impl, allegedly numerically stable
    neg_abs = -jnp.abs(logit_in)
    loss = jnp.clip(logit_in, a_min=0) - logit_in * target + jnp.log(1 + jnp.exp(neg_abs))
    return loss

# interval routines
def smallest_magnitude(interval_lower, interval_upper):
    min_mag = jnp.maximum(jnp.abs(interval_lower), jnp.abs(interval_upper))
    min_mag = jnp.where(jnp.logical_and(interval_upper > 0, interval_lower < 0), 0., min_mag)
    return min_mag
    
def biggest_magnitude(interval_lower, interval_upper):
    return jnp.maximum(interval_upper, -interval_lower)


def sin_bound(lower, upper):
    '''
    Bound sin([lower,upper])
    '''
    f_lower = jnp.sin(lower)
    f_upper = jnp.sin(upper)

    # test if there is an interior peak in the range
    lower /= 2. * jnp.pi
    upper /= 2. * jnp.pi
    contains_min = jnp.ceil(lower - .75) < (upper - .75)
    contains_max = jnp.ceil(lower - .25) < (upper - .25)

    # result is either at enpoints or maybe an interior peak
    out_lower = jnp.minimum(f_lower, f_upper)
    out_lower = jnp.where(contains_min, -1., out_lower)
    out_upper = jnp.maximum(f_lower, f_upper)
    out_upper = jnp.where(contains_max, 1., out_upper)

    return out_lower, out_upper

def cos_bound(lower, upper):
    return sin_bound(lower + jnp.pi/2, upper + jnp.pi/2)
