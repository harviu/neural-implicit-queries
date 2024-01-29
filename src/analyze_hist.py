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

def draw(mu, sigma, Z, ax, label):
    x = np.linspace(mu - sigma * (Z+1), mu+sigma*(Z+1), 100)
    y = norm.pdf(x, mu, sigma)
    ax.plot(x,y,label=label)


fig, ax_array = plt.subplots(4,3, figsize=(7.5,8))
data_opts = ['Vortex', 'Asteroid', 'Combustion', 'Ethanediol','Isotropic','fox', 'hammer','birdcage','bunny']
for i, data_type in enumerate([0, 3, 2, 4]):
    ax_array[i][0].set_ylabel(data_opts[data_type])
    if data_type == 0:
        # test_model = 'sample_inputs/vorts_sin_3_128.npz'
        test_model = 'sample_inputs/vorts_sin_8_32.npz'
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

    N = 5
    dkl1 = 0
    dkl2 = 0
    dkl3 = 0
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    implicit_func1, params1 = generate_implicit_from_file(test_model, mode="uncertainty_all")
    implicit_func2, params2 = generate_implicit_from_file(test_model, mode="affine_ua")

    #uniform
    # rand_n = jax.random.uniform(subkey,(N,3))
    # rand_n = (rand_n - 0.5) * 2 #* 0.95
    # if data_type in [0,3]:
    #     scale = jnp.array((0.05,0.05,0.05))
    # else:
    #     scale = jnp.array((0.02,0.02,0.02))
    # scale = jnp.array((20) * 3)
    x_label= ['Small', 'Median', 'Large']
    scale_size = [0.1, 1, 10]
    center = jnp.array((0.5) * 3)
    # analyze histogram
    for j, scale_s in enumerate(scale_size):
        ax_array[-1][j].set_xlabel('Test')
        scale = jnp.array((scale_s,) * 3)
        range_lower = center - scale
        range_higher = center + scale

        #MC
        vals, mu, sigma = compare_mc_clt(implicit_func1, params1, range_lower, range_higher, n=1e6)
        vals = np.asarray(vals)
        counts, bins = np.histogram(vals, 100,density=True)

        #UP
        dkl1 += kl(counts, bins, mu, sigma)
        
        #RAUA
        samples, mu_raua, sigma_raua = compare_mc_clt(implicit_func2, params2, range_lower, range_higher, n=50)
        dkl2 += kl(counts, bins, mu_raua, sigma_raua)

        
        # sample 100 values and fit Gaussian using MLE
        mu_sample, sigma_sample = norm.fit(samples)
        dkl3 += kl(counts, bins, mu_sample, sigma_sample)

        ax = ax_array[i][j]
        ax.stairs(counts, bins, label='MC')
        draw(mu, sigma, 2, ax, 'UP')
        # draw(mu_raua, sigma_raua, 2, ax,'RA-UA')
        draw(mu_sample, sigma_sample, 3, ax,'SAMPLE')
        # bin_range = bins[-1] - bins[0]
        # print(bins[0], bins[-1])
        # ax.set_xlim(bins[0]-bin_range/2,bins[-1]+bin_range/2)
        # ax.set_xlim(bins[0],bins[-1])
        # plt.savefig('hist_compare/{}_{}_{}.png'.format(data_opts[data_type],range_lower,range_higher))
        print(dkl1, dkl2, dkl3)
    print(dkl1/N, dkl2/N, dkl3/N)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.875,0.82))
fig.tight_layout()
plt.savefig('hist_compare/dist.png', bbox_inches='tight')
plt.savefig('hist_compare/dist.pdf', bbox_inches='tight')