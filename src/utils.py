import time
import os
import inspect

from functools import partial

import jax
import jax.numpy as jnp
import datetime

import numpy as np

import polyscope as ps
import polyscope.imgui as psim
from vtkmodules import all as vtk
from vtkmodules.util import numpy_support
import netCDF4 as nc


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

def load_data(data_type, data_path):
    
    if data_type == 'vorts':
        data = load_vorts_data(128, data_path)
    elif data_type == 'asteroid':
        data = load_asteroid_data(500, data_path)
    elif data_type == 'combustion':
        data = load_combustion_data(data_path)
    elif data_type == 'ethanediol':
        data = load_ethanediol_data(data_path)
    elif data_type == 'isotropic':
        data = load_isotropic_data(data_path)
    else:
        raise ValueError("not implemented data type")
    return data

def load_isotropic_data(data_path):
    ds = nc.Dataset(data_path)
    numpy_array = ds['a'][:]
    return jnp.asarray(numpy_array)

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
    def __init__(self, name=None, filename=None, warmup = False):
        self.name = name 
        self.filename = filename
        self.warmup = warmup

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.3f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        if not self.warmup: print(message)
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
