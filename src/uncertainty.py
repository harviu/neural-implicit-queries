from functools import partial
import dataclasses 
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

import jax
import jax.numpy as jnp

import utils

import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

# === Function wrappers

class UncertaintyImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, func, ctx):
        super().__init__("classify-only")
        self.affine_func = func
        self.ctx = ctx
        self.mode_dict = {'ctx' : self.ctx}


    def __call__(self, params, x):
        f = lambda x : self.affine_func(params, x, self.mode_dict)
        return wrap_scalar(f)(x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_general_box(self, params, box_center, box_vecs, isovalue=0., offset=0., z = 2):

        d = box_center.shape[-1]
        v = box_vecs.shape[-2]
        assert box_center.shape == (d,), "bad box_vecs shape"
        assert box_vecs.shape == (v,d), "bad box_vecs shape"
        keep_ctx = dataclasses.replace(self.ctx, affine_domain_terms=v)

        # evaluate the function
        input = coordinates_in_general_box(keep_ctx, box_center, box_vecs)
        output = self.affine_func(params, input, {'ctx' : keep_ctx})

        # compute relevant bounds
        # compute the justified prob_threshold
        may_lower, may_upper = may_contain_bounds(keep_ctx, output, z=z)

        # determine the type of the region
        output_type = SIGN_UNKNOWN
        output_type = jnp.where(may_lower > isovalue+offset, SIGN_POSITIVE, output_type)
        output_type = jnp.where(may_upper < isovalue-offset, SIGN_NEGATIVE, output_type)

        return output_type

    # def estimate_general_box_bounds(self, params, box_center, box_vecs):
    #     d = box_center.shape[-1]
    #     v = box_vecs.shape[-2]
    #     assert box_center.shape == (d,), "bad box_vecs shape"
    #     assert box_vecs.shape == (v,d), "bad box_vecs shape"
    #     keep_ctx = dataclasses.replace(self.ctx, affine_domain_terms=v)

    #     # evaluate the function
    #     input = coordinates_in_general_box(keep_ctx, box_center, box_vecs)
    #     output = self.affine_func(params, input, {'ctx' : keep_ctx})
    #     base, aff, err = output

    #     # compute relevant bounds
    #     may_lower, may_upper = may_contain_bounds(keep_ctx, output)
    #     return may_lower, may_upper, base, aff, err # return output for debug reason

# === Affine utilities

# We represent affine data as a tuple input=(base,aff,err). Base is a normal shape (d,) primal vector value, affine is a (v,d) array of affine coefficients (may be v=0), err is a centered interval error shape (d,), which must be nonnegative.
# For constant values, aff == err == None. If is_const(input) == False, then it is guaranteed that aff and err are non-None.

@dataclass(frozen=True)
class AffineContext():
    mode: str = 'uncertainty_all'
    truncate_count: int = -777
    truncate_policy: str = 'absolute'
    affine_domain_terms: int = 0
    n_append: int = 0

    def __post_init__(self):
        if self.mode not in ['uncertainty_truncate', 'uncertainty_all']:
            raise ValueError("invalid mode")

        if self.mode == 'uncertainty_truncate':
            if self.truncate_count is None:
                raise ValueError("must specify truncate count")

def is_const(input):
    mu, vecs, sigma, err = input
    if err is not None: 
        return False
    if sigma is not None:
        if sigma.shape[0] > 0: return False
    if vecs is not None:
        if vecs.shape[0] > 0: return False
    return True


# Compute the mu and sigma 
def radius(input):
    if is_const(input): return 0.
    mu, vecs, sigma, err = input
    vecs_var = ((vecs * vecs) / 3).sum(0)
    var = (sigma * sigma).sum(0)
    var_sum = var + vecs_var
    if err is not None:
        var_sum += err * err
    return mu, jnp.sqrt(var_sum)

# Constuct affine inputs for the coordinates in k-dimensional box,
# which is not necessarily axis-aligned
#  - center is the center of the box
#  - vecs is a (V,D) array of vectors which point from the center of the box to its
#    edges. These will correspond to each of the affine symbols, with the direction
#    of the vector becoming the positive orientaiton for the symbol.
# (this function is nearly a no-op, but giving it this name makes it easier to
#  reason about)
def coordinates_in_general_box(ctx, center, vecs):
    base = center
    aff = jnp.zeros((0,center.shape[-1]))
    err = jnp.zeros_like(center)
    return base, vecs, aff, err


# implemented clt in radius function
def may_contain_bounds(ctx, input, z=2):
    mu, vecs, sigma, err = input
    vecs_bound = jnp.sum(jnp.abs(vecs), axis=0) # use exact bounds for the vectors
    variance = jnp.sum(sigma ** 2, axis=0) # calculat the variance for uncertainty terms
    if err is not None:
        variance += err * err # add the truncation error
    sigma = jnp.sqrt(variance)
    z_bound = z * sigma + vecs_bound
    return mu - z_bound, mu + z_bound

def truncate_affine(ctx, input):
    # do nothing if the input is a constant or we are not in truncate mode
    if is_const(input): return input
    if ctx.mode != 'uncertainty_truncate':
        return input

    # gather values
    # only apply the truncation to sigma
    mu, vecs, sigma, err = input
    n_keep = ctx.truncate_count

    # if the affine list is shorter than the truncation length, nothing to do
    if sigma.shape[0] <= n_keep:
        return input

    # compute the magnitudes of each affine value
    # TODO fanicier policies?
    if ctx.truncate_policy == 'absolute':
        affine_mags = jnp.sum(jnp.abs(sigma), axis=-1)
    elif ctx.truncate_policy == 'relative':
        affine_mags = jnp.sum(jnp.abs(sigma / mu[None,:]), axis=-1)
    else:
        raise RuntimeError("bad policy")


    # sort the affine terms by by magnitude
    sort_inds = jnp.argsort(-affine_mags, axis=-1) # sort to decreasing order
    sigma = sigma[sort_inds,:]

    # keep the n_keep highest-magnitude entries
    aff_keep = sigma[:n_keep,:]

    # for all the entries we aren't keeping, add their contribution to the interval error
    aff_drop = sigma[n_keep:,:]
    err = err * err + jnp.sum((aff_drop * aff_drop), axis=0)
    err = jnp.sqrt(err)

    return mu, vecs, aff_keep, err

def apply_linear_approx(ctx, input, alpha, beta, delta):
    mu, vecs, sigma, err = input
    mu = alpha * mu + beta
    if sigma is not None:
        sigma = alpha * sigma
    if vecs is not None:
        vecs = alpha * vecs
    if err is not None:
        err = alpha * err

    # This _should_ always be positive by definition. Always be sure your 
    # approximation routines are generating positive delta.
    # At most, we defending against floating point error here.
    delta = jnp.abs(delta)

    new_sigma = jnp.diag(delta)
    sigma = jnp.concatenate((sigma, new_sigma), axis=0)
    return truncate_affine(ctx, (mu, vecs, sigma, err))

# Convert to/from the affine representation from an ordinary value representing a scalar
def from_scalar(x):
    return x, None, None, None
def to_scalar(input):
    if not is_const(input):
        raise ValueError("non const input")
    return input[0]
def wrap_scalar(func):
    return lambda x : to_scalar(func(from_scalar(x)))
