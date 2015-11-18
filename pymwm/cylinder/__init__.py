# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import jv, jvp, kv, kvp, jn_zeros, jnp_zeros


class Cylinder(object):
    """A class defining a cylindrical waveguide.

     Attributes:
        fill: An instance of Material class for the core
        clad: An instance of Material class for the clad
        r: A float indicating the radius of the circular cross section [um].
        samples: An instance of Samples class
    """

    def __init__(self, params):
        """Init Cylinder class.

        Args:
            params: A dict whose keys and values are as follows:
                'core': A dict of the setting parameters of the core:
                    'shape': A string indicating the shape of the core.
                    'size': A float indicating the radius of the circular cross
                        section [um].
                    'fill': A dict of the parameters of the core Material.
                'clad': A dict of the parameters of the clad Material.
                'bounds': A dict indicating the bounds of interpolation, and
                    its keys and values are as follows:
                    'lmax': A float indicating the maximum wavelength [um]
                    'lmin': A float indicating the minimum wavelength [um]
                    'limag': A float indicating the minimum value of
                        abs(c / fimag) [um] where fimag is the imaginary part
                        of the frequency.
                'modes': A dict of the settings for calculating modes:
                    'lmax': A float indicating the maximum wavelength [um]
                        (defulat: 5.0)
                    'lmin': A float indicating the minimum wavelength [um]
                        (defulat: 0.4)
                    'limag': A float indicating the minimum value of
                        abs(c / fimag) [um] where fimag is the imaginary part
                        of the frequency. (default: 5.0)
                    'dw': A float indicating frequency interval
                        [rad c / 1um]=[2.99792458e14 rad / s]
                        (default: 1 / 64).
                    'num_n': An integer indicating the number of orders of
                        modes.
                    'num_m': An integer indicating the number of modes in each
                        order and polarization.
        """
        from pyoptmat import Material
        from pymwm.cylinder.samples import Samples
        self.r = params['core']['size']
        self.fill = Material(params['core']['fill'])
        self.clad = Material(params['clad'])
        self.samples = Samples(self.r, self.fill, self.clad, params['modes'])
        self.bounds = params['bounds']
        try:
            betas, convs = self.samples.load()
        except KeyError:
            from multiprocessing import Pool
            num_n = params['modes']['num_n']
            p = Pool(num_n)
            betas_list = p.map(self.samples, range(num_n))
            betas = {key: val for betas, convs in betas_list
                     for key, val in betas.items()}
            convs = {key: val for betas, convs in betas_list
                     for key, val in convs.items()}
            self.samples.save(betas, convs)
        self.beta_funcs = self.samples.interpolation(betas, convs, self.bounds)
