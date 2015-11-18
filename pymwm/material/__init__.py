# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from scipy.constants import c


class Material(object):

    """A class defining the dielectric function for a material.

    Attributes:
        model: A string indicating the model of dielectric function.
    """

    def __init__(self, params):
        """Init Material class.

        Args:
            params: A dict whose keys are as follows.
                'model': A string indicating the model of dielectric function,
                    that must be 'dielectric', 'drude', 'drude_lorentz',
                    'haftel', 'gold_d', 'gold_dl', 'gold_h', 'air' or 'water'.
                    (default 'dielectric')
                'e': For 'dielectric'
                'e', 'wp', 'gp': For 'drude'
                'e1', 'e1', 'tau', 'sp': For 'haftel'.
                'e', 'wp', 'gp', 'ss', 'ws', 'gs': For 'drude_lorentz'.
        """
        c0 = c * 1e-8
        self.__w = None
        self.__eps = None
        self._params = params
        model = self._params['model']
        self.im_factor = self._params.get('im_factor', 1.0)
        self.model = model
        self.RIs = {
            'air': 1.0,
            'water': 1.333,
            'SiN': 2.0,
            'KRS5': 2.4,
            'ZnSe': 2.4,
            'Ge': 4.092,
            'Si': 3.4179,
            'FK03': 1.43875,
            'FK02': 1.456,
            'FK01': 1.497,
            'FK1': 1.497,
            'FK1': 1.4706,
            'FK5': 1.4874,
            'FK8': 1.51118,
            'FK3': 1.51454,
            'FK6': 1.51742,
            'BK7': 1.51633,
            'BaK2': 1.53996,
            'BaK4': 1.56883,
            'BaK1': 1.5725,
            'SF7': 1.6398,
            'SF2': 1.64769,
            'SF19': 1.6668,
            'SF5': 1.6727,
            'SF8': 1.68893,
            'SF15': 1.69895,
            'SF1': 1.71736,
            'SF18': 1.72151,
            'SF10': 1.72825,
            'SF13': 1.74077,
            'SF4': 1.7552,
            'SF14': 1.76182,
            'SF11': 1.78472,
            'SF6': 1.80518,
            'SF03': 1.84666}
        if 'gold_jc' in model:
            from pymwm.material.jc import Johnson_Christy
            self._johnson_christy = Johnson_Christy('gold')
        elif 'silver_jc' in model:
            from pymwm.material.jc import Johnson_Christy
            self._johnson_christy = Johnson_Christy('silver')
        elif 'copper_jc' in model:
            from pymwm.material.jc import Johnson_Christy
            self._johnson_christy = Johnson_Christy('copper')
        if 'gold_palik' in model:
            from pymwm.material.palik import Palik
            self._palik = Palik('gold')
        elif 'silver_palik' in model:
            from pymwm.material.palik import Palik
            self._palik = Palik('silver')
        elif 'copper_palik' in model:
            from pymwm.material.palik import Palik
            self._palik = Palik('copper')
        elif 'gold_dl' in model:
            self._params['e'] = 5.3983
            self._params['wp'] = 13.978 / c0 * 10
            self._params['gp'] = 1.0334e-1 / c0 * 10
            self._params['ss'] = (2.5417 * 0.2679, 2.5417 * 0.7321)
            self._params['ws'] = (4.2739 / c0 * 10, 5.2254 / c0 * 10)
            self._params['gs'] = (2 * 4.3533e-1 / c0 * 10,
                                  2 * 6.6077e-1 / c0 * 10)
        elif 'silver_dl' in model:
            self._params['e'] = 1.7984
            self._params['wp'] = 13.359 / c0 * 10
            self._params['gp'] = 0.087167 / c0 * 10
            self._params['ss'] = (3.0079, 2.3410)
            self._params['ws'] = (8.1635 / c0 * 10, 38.316 / c0 * 10)
            self._params['gs'] = (437.85 / c0 * 10, 60.574 / c0 * 10)
        elif 'gold_d' in model:
            self._params['e'] = 9.0685
            self._params['wp'] = 2 * np.pi * 2.1556 / c0 * 10
            self._params['gp'] = 2 * np.pi * 1.836e-2 / c0 * 10
        elif 'silver_d' in model:
            self._params['e'] = 4.0
            self._params['wp'] = 13.7 / c0 * 10
            self._params['gp'] = 2 * np.pi * 8.5e-2 / c0 * 10
        elif model == 'polymer':
            self._params['e'] = 2.26
        elif model == 'solution':
            self._params['e'] = self._params['RI'] ** 2
        elif model in self.RIs:
            self._params['e'] = self.RIs[model] ** 2

    def __call__(self, w):
        """Return a float indicating the permittivity.

        Args:
            w: A float indicating the angular frequency.

        Raises:
            ValueError: The model is not defined.
        """
        if self.__w is None or w != self.__w:
            self.__w = w
            model = self.model
            p = self._params
            if model in self.RIs or model == 'polymer' or model == 'solution':
                self.__eps = p['e']
            elif model[-3:] == '_jc':
                self.__eps = self._johnson_christy(w.real)
            elif model[-6:] == '_palik':
                self.__eps = self._palik(w.real)
            elif model[-3:] == '_dl':
                self.__eps = self._drude_lorentz(w, p)
            elif model[-2:] == '_d':
                self.__eps = self._drude(w, p)
            elif model[-2:] == '_h':
                self.__eps = self._haftel(w, p)
            elif model == 'pec':
                self.__eps = -1e8
            elif model == 'drude':
                self.__eps = self._drude(w, p)
            elif model == 'haftel':
                self.__eps = self._haftel(w, p)
            elif model == 'drude_lorentz':
                self.__eps = self._drude_lorentz(w, p)
            else:
                raise ValueError("The model is not defined.")
            if 'no_loss' in model:
                self.__eps = self.__eps.real + self.__eps.imag * 1e-6j
        return self.__eps

    def _drude(self, w, p):
        return p['e'] - p['wp'] ** 2 / (w ** 2 +
                                        1.0j * self.im_factor * p['gp'] * w)

    def _drude_lorentz(self, w, p):
        eps = p['e'] - p['wp'] ** 2 / (w ** 2 + 1.0j * p['gp'] * w)
        for sn, wn, gn in zip(p['ss'], p['ws'], p['gs']):
            eps -= sn * wn ** 2 / (w ** 2 - wn ** 2 +
                                   1.0j * self.im_factor * gn * w)
        return eps
