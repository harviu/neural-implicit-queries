from functools import partial
import dataclasses 
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

import utils

import implicit_function
from implicit_function import SIGN_UNKNOWN, SIGN_POSITIVE, SIGN_NEGATIVE

# === Function wrappers

class MCImplicitFunction(implicit_function.ImplicitFunction):

    def __init__(self, func, lipschitz_bound=1.):
        super().__init__("classify-only")
        self.func = func

    def __call__(self, params, x):
        return self.func(params, x)

    # the parent class automatically delegates to this
    # def classify_box(self, params, box_lower, box_upper):
        # pass
        
    def classify_box(self, params, box_lower, box_upper, isovalue=0., offset=0., prob_threshold = 2., num_grid=0):

        # uniform sample in the box
        n = 100
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        #uniform
        rand_n = jax.random.uniform(subkey,(n,3))
        delta = box_upper - box_lower
        coords = box_lower + delta * rand_n
        vals = jax.vmap(partial(self.func, params))(coords)

        mean = jnp.mean(vals)
        sigma = jnp.sqrt(jnp.var(vals))

        # determine the type of the region
        output_type = SIGN_UNKNOWN
        output_type = jnp.where(mean - sigma * prob_threshold > isovalue+offset, SIGN_POSITIVE, output_type)
        output_type = jnp.where(mean + sigma * prob_threshold < isovalue-offset, SIGN_NEGATIVE, output_type)
        return output_type
