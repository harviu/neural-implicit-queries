from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import jax.numpy as jnp
import jax
from evaluation import compare_mc_clt
from implicit_mlp_utils import generate_implicit_from_file

def kl(counts, bins, mu, sigma):
    centers = (bins[1:] + bins[:-1])/2
    diff = (bins[1:] - bins[:-1])
    px = counts + 1e-20
    qx = norm.pdf(centers, mu, sigma) + 1e-20
    kl1 = px * diff * np.log(px/qx)
    kl1 = kl1.sum()
    return kl1


key = jax.random.PRNGKey(42)

data_opts = ['Vortex', 'Asteroid', 'Combustion', 'Ethanediol','Isotropic','fox', 'hammer','birdcage','bunny']
for i, data_type in enumerate([0]):
    if data_type == 0:
        test_model = 'sample_inputs/vorts_sin_8_32.npz'
        input_file = '../data/vorts01.data'
        bounds = np.array([127, 127, 127])
        isovalue = 2
    elif data_type == 1:
        test_model = 'sample_inputs/v02_relu_8_32.npz'
        input_file = '../data/99_500_v02.bin'
        bounds = np.array([499, 499, 499])
    elif data_type == 2:
        test_model = 'sample_inputs/jet_sin_8_32.npz'
        input_file = '../data/jet_chi_0054.dat'
        bounds = np.array([479, 339, 119])
    elif data_type == 3:
        test_model = 'sample_inputs/eth_sin_8_32.npz'
        input_file = '../data/ethanediol.bin'
        bounds = np.array([115, 116, 134])
        isovalue = -2.2
    elif data_type == 4:
        test_model = 'sample_inputs/iso_sin_3_128.npz'
        input_file = '../data/Isotropic.nz'
        bounds = np.array([1024,1024,1024])
        isovalue = 0
    elif data_type == 5:
        test_model = 'sample_inputs/fox_relu.npz'
    elif data_type == 6:
        test_model = 'sample_inputs/hammer.npz'
    elif data_type == 7:
        test_model = 'sample_inputs/birdcage_occ.npz'
    elif data_type == 8:
        test_model = 'sample_inputs/bunny.npz'

    N = 4000
    valid = 0
    dkl1 = 0
    dkl2 = 0
    dkl3 = 0

    implicit_func1, params1 = generate_implicit_from_file(test_model, mode="uncertainty_all")
    implicit_func2, params2 = generate_implicit_from_file(test_model, mode="affine_ua")

    #uniform
    key, subkey = jax.random.split(key)
    rand_n = jax.random.uniform(subkey,(N,3))
    center = (rand_n - 0.5) * 2
    # if data_type in [0,3]:
    #     scale = jnp.array((0.05,0.05,0.05))
    # else:
    #     scale = jnp.array((0.02,0.02,0.02))
    key, subkey = jax.random.split(key)
    scale = jax.random.uniform(subkey,(N,1))
    lower = center - scale  
    upper = center + scale
    # analyze histogram
    for j in range(N):
        range_lower = lower[j]
        range_higher = upper[j]

        if range_lower.min() < -1:
            continue
        if range_higher.max() > 1:
            continue

        valid += 1

        #MC
        vals, mu, sigma = compare_mc_clt(implicit_func1, params1, range_lower, range_higher, n=1e6)
        vals = np.asarray(vals)
        # vals = vals * 100
        # mu *= 100
        # sigma *= 100
        counts, bins = np.histogram(vals, 100,density=True)

        #UP
        dkl1 += kl(counts, bins, mu, sigma)
        
        #RAUA
        samples, mu_raua, sigma_raua = compare_mc_clt(implicit_func2, params2, range_lower, range_higher, n=10)
        # samples = samples * 100
        # mu_raua *= 100
        # sigma_raua *= 100
        dkl2 += kl(counts, bins, mu_raua, sigma_raua)

        # sample 100 values and fit Gaussian using MLE
        mu_sample, sigma_sample = norm.fit(samples)
        dkl3 += kl(counts, bins, mu_sample, sigma_sample)

    print(valid)
    print(dkl1/valid, dkl2/valid, dkl3/valid)
