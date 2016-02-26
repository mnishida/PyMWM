import matplotlib.pyplot as plt
import numpy as np
from pymwm.waveguide import create
from multiprocessing import Pool
params = {'core': {'shape': 'slit', 'size': 0.15,
                   'fill': {'model': 'air'}},
          'clad': {'model': 'gold_dl'},
          'modes': {'lmax': 5.0, 'lmin': 0.55, 'limag': 5.0,
                    'nwr': 512, 'nwi': 64, 'num_n': 6,
                    'num_m': 2}}
num_n = params['modes']['num_n']
wg = create(params)
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
