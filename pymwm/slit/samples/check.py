import matplotlib.pyplot as plt
import numpy as np
from pyoptmat import Material
from pymwm.slit.samples import Samples
from multiprocessing import Pool
params = {'core': {'shape': 'slit', 'size': 0.3,
                   'fill': {'model': 'air'}},
          'clad': {'model': 'gold_dl'},
          'modes': {'lmax': 1.2, 'lmin': 0.545, 'limag': 5.0,
                    'dw': 1.0 / 64,
                    'num_n': 6, 'num_m': 2}}
r = params['core']['size']
fill = Material(params['core']['fill'])
clad = Material(params['clad'])
num_n = params['modes']['num_n']
wg = Samples(r, fill, clad, params['modes'])
p = Pool(num_n)
betas_list = p.map(wg, range(num_n))
betas = {key: val for betas, convs in betas_list
         for key, val in betas.items()}
convs = {key: val for betas, convs in betas_list
         for key, val in convs.items()}


def plot_convs(pol, n, m):
    X, Y = np.meshgrid(wg.ws, wg.wis, indexing='ij')
    Z = convs[(pol, n, m)]
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.show()


def plot_real_betas(pol, n, m):
    X, Y = np.meshgrid(wg.ws, wg.wis, indexing='ij')
    Z = betas[(pol, n, m)]
    plt.pcolormesh(X, Y, Z.real)
    plt.colorbar()
    plt.show()


def plot_imag_betas(pol, n, m):
    X, Y = np.meshgrid(wg.ws, wg.wis, indexing='ij')
    Z = betas[(pol, n, m)]
    plt.pcolormesh(X, Y, Z.imag)
    plt.colorbar()
    plt.show()
