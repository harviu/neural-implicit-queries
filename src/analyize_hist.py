from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import jax.numpy as jnp
import jax
from evaluation import compare_mc_clt
from implicit_mlp_utils import generate_implicit_from_file


fig, ax_array = plt.subplots(4,5, figsize=(15,8))
data_opts = ['Vortex', 'Asteroid', 'Combustion', 'Ethanediol','Isotropic','fox', 'hammer','birdcage','bunny']
for i, data_type in enumerate([0,2,3,4]):
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
        test_model = 'sample_inputs/fox.npz'
    elif data_type == 6:
        test_model = 'sample_inputs/hammer.npz'
    elif data_type == 7:
        test_model = 'sample_inputs/birdcage_occ.npz'
    elif data_type == 8:
        test_model = 'sample_inputs/bunny.npz'

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    #uniform
    rand_n = jax.random.uniform(subkey,(5,3))
    rand_n = (rand_n - 0.5) * 0.95 * 2

    center = jnp.array((0,0,0))
    if data_type in [0,3]:
        scale = jnp.array((0.05,0.05,0.05))
    else:
        scale = jnp.array((0.02,0.02,0.02))
    # analyze histogram
    for j, center in enumerate(rand_n):
        Z = 2
        ax = ax_array[i][j]
        range_lower = center - scale
        range_higher = center + scale

        mode = "uncertainty_all"
        implicit_func, params = generate_implicit_from_file(test_model, mode=mode)
        vals, mu, sigma = compare_mc_clt(implicit_func, params, range_lower, range_higher, n=1e6)
        vals = np.asarray(vals)
        # print('min, max:', vals.min(),vals.max())
        # print('mean:', vals.mean())
        counts, bins = np.histogram(vals, 100,density=True)
        ax.stairs(counts, bins, label='MC')
        # plt.hist(vals, 100, density=True)
        x = np.linspace(mu - sigma * (Z+1), mu+sigma*(Z+1), 100)
        y = norm.pdf(x, mu, sigma)
        # plt.xlim([x[0],x[-1]])
        ax.plot(x,y,label='UP')
        
        mode = "affine_ua"
        implicit_func, params = generate_implicit_from_file(test_model, mode=mode)
        vals, mu, sigma = compare_mc_clt(implicit_func, params, range_lower, range_higher)
        x = np.linspace(mu - sigma * Z, mu+sigma*Z, 100)
        y = norm.pdf(x, mu, sigma)
        ax.plot(x,y,label='RAUA')

        # plt.savefig('hist_compare/{}_{}_{}.png'.format(data_opts[data_type],range_lower,range_higher))
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.875,0.82))
# fig.tight_layout()
plt.savefig('hist_compare/dist.png', bbox_inches='tight')
plt.savefig('hist_compare/dist.pdf', bbox_inches='tight')