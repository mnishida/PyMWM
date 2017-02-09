#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Dict, Tuple
import abc
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas import DataFrame
from pyoptmat import Material


class Waveguide(metaclass=abc.ABCMeta):
    """A class defining a waveguide."""

    @abc.abstractproperty
    def fill(self) -> Material:
        pass

    @abc.abstractproperty
    def clad(self) -> Material:
        pass

    @abc.abstractproperty
    def r(self) -> float:
        pass

    @abc.abstractproperty
    def samples(self):
        pass

    @abc.abstractproperty
    def fill(self):
        pass

    @abc.abstractproperty
    def clad(self):
        pass


class Sampling(metaclass=abc.ABCMeta):
    """A class provides sampling methods."""

    @abc.abstractmethod
    def __init__(self):
        self.shape = None
        self.r = None
        self.num_all = None

    @abc.abstractproperty
    def fill(self):
        pass

    @abc.abstractproperty
    def clad(self):
        pass

    @abc.abstractproperty
    def params(self):
        pass

    @property
    def key(self) -> Dict:
        p = self.params
        dw = p.setdefault('dw', 1.0 / 64)
        lmax = p.setdefault('lmax', 5.0)
        lmin = p.setdefault('lmin', 0.4)
        limag = p.setdefault('limag', 5.0)
        shape = self.shape
        size = self.r
        if self.fill.model == 'dielectric':
            core = "RI_{}".format(self.fill.params['RI'])
        else:
            core = "{0}".format(self.fill.model)
        if self.clad.model == 'dielectric':
            clad = "RI_{}".format(self.clad.params['RI'])
        else:
            clad = "{0}".format(self.clad.model)
        num_n = p['num_n']
        num_m = p['num_m']
        num_all = self.num_all
        im_factor = self.clad.im_factor
        d = dict((
            ('shape', shape), ('size', size), ('core', core), ('clad', clad),
            ('lmax', lmax), ('lmin', lmin), ('limag', limag),
            ('dw', dw), ('num_n', num_n), ('num_m', num_m),
            ('num_all', num_all), ('im_factor', im_factor)))
        return d


class Database:
    """The interface with the database of propagation constants."""

    dirname = os.path.join(os.path.expanduser('~'), '.pymwm')
    filename = os.path.join(dirname, 'pymwm_data.h5')
    catalog_columns = OrderedDict((
        ('id', int), ('shape', str), ('size', float), ('core', str),
        ('clad', str),
        ('lmax', float), ('lmin', float), ('limag', float), ('dw', float),
        ('num_n', int), ('num_m', int), ('im_factor', float), ('EM', str),
        ('n', int), ('m', int)))
    data_columns = OrderedDict((
        ('conv', bool), ('beta_real', float), ('beta_imag', float)))

    def __init__(self, key: Dict):
        if not os.path.exists(self.filename):
            if not os.path.exists(self.dirname):
                os.mkdir(self.dirname)
            store = pd.HDFStore(self.filename, complevel=9, complib='blosc')
            catalog = pd.DataFrame(columns=self.catalog_columns.keys())
            store['catalog'] = catalog
            store.close()
        self.shape = key['shape']
        self.size = key['size']
        self.core = key['core']
        self.clad = key['clad']
        self.lmax = key['lmax']
        self.lmin = key['lmin']
        self.limag = key['limag']
        self.dw = key['dw']
        self.num_n = key['num_n']
        self.num_m = key['num_m']
        self.num_all = key['num_all']
        self.im_factor = key['im_factor']
        ind_wmin = int(np.floor(2 * np.pi / self.lmax / self.dw))
        ind_wmax = int(np.ceil(2 * np.pi / self.lmin / self.dw))
        ind_wimag = int(np.ceil(2 * np.pi / self.limag / self.dw))
        self.ws = np.arange(ind_wmin, ind_wmax + 1) * self.dw
        self.wis = -np.arange(ind_wimag + 1) * self.dw

    @property
    def id(self) -> int:
        with pd.HDFStore(self.filename, "r") as store:
            try:
                catalog = store['catalog']
            except KeyError:
                return 0
        cond = ((catalog['shape'] == self.shape) &
                (catalog['size'] == self.size) &
                (catalog['core'] == self.core) &
                (catalog['clad'] == self.clad) &
                (catalog['lmax'] == self.lmax) &
                (catalog['lmin'] == self.lmin) &
                (catalog['limag'] == self.limag) &
                (catalog['dw'] == self.dw) &
                (catalog['num_n'] == self.num_n) &
                (catalog['num_m'] == self.num_m) &
                (catalog['im_factor'] == self.im_factor))
        ids = catalog[cond].index
        if len(ids) != self.num_all:
            raise Exception("Database is broken.")
        if len(ids):
            return ids[0]
        else:
            return len(catalog.index)

    @staticmethod
    def set_columns_dtype(df: DataFrame, columns: Dict):
        """Set data type of each column in the DataFrame."""
        for key, val in columns.items():
            df[key] = df[key].astype(val)

    def load(self) -> Tuple[Dict, Dict]:
        num_wr = len(self.ws)
        num_wi = len(self.wis)
        with pd.HDFStore(self.filename, "r") as store:
            betas = dict()
            convs = dict()
            for idx in range(self.id, self.id + self.num_all):
                catalog = store['catalog']
                em = catalog.loc[idx, 'EM']
                n = catalog.loc[idx, 'n']
                m = catalog.loc[idx, 'm']
                data = store['/id_{}'.format(idx)]
                conv = data['conv']
                beta_real = data['beta_real']
                beta_imag = data['beta_imag']
                convs[(em, n, m)] = conv.values.reshape(num_wr, num_wi)
                beta = np.zeros_like(beta_real.values, dtype=np.complex128)
                beta.real = beta_real.values
                beta.imag = beta_imag.values
                betas[(em, n, m)] = beta.reshape(num_wr, num_wi)
        return betas, convs

    def save(self, betas: Dict, convs: Dict):
        with pd.HDFStore(self.filename, complevel=9, complib='blosc') as store:
            catalog = store['catalog']
            idx = self.id
            for EM, n, m in sorted(convs.keys()):
                se = pd.Series([idx, self.shape, self.size, self.core,
                                self.clad, self.lmax, self.lmin, self.limag,
                                self.dw, self.num_n, self.num_m, self.im_factor,
                                EM, n, m], index=self.catalog_columns.keys())
                catalog = catalog.append(se, ignore_index=True)
                conv = convs[(EM, n, m)].ravel()
                beta = betas[(EM, n, m)].ravel()
                df = pd.DataFrame(
                    {'conv': conv, 'beta_real': beta.real,
                     'beta_imag': beta.imag},
                    columns=self.data_columns.keys())
                self.set_columns_dtype(df, self.data_columns)
                store.append('id_{}'.format(idx), df)
                idx += 1
            self.set_columns_dtype(catalog, self.catalog_columns)
            catalog.set_index('id', inplace=True)
            store['catalog'] = catalog

    def delete(self):
        with pd.HDFStore(
                self.filename, complevel=9, complib='blosc') as store:
            catalog = store['catalog']
            for idx in range(self.id, self.id + self.num_all):
                catalog.drop(idx)
                store.remove("id_{}".format(idx))
            store['catalog'] = catalog
        os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 " +
                  "--complib=blosc {0} {0}.new".format(self.filename))
        os.system("mv {0}.new {0}".format(self.filename))

    def interpolation(self, betas: np.ndarray, convs: np.ndarray,
                      bounds: Dict) -> Dict:
        from scipy.interpolate import RectBivariateSpline
        lmax = bounds['lmax']
        lmin = bounds['lmin']
        limag = bounds['limag']
        wr_min = 2 * np.pi / lmax
        wr_max = 2 * np.pi / lmin
        wi_min = -2 * np.pi / limag
        num_n = self.num_n
        num_m = self.num_m
        ws = self.ws
        wis = self.wis[::-1]
        imin = np.searchsorted(ws, [wr_min], side='right')[0] - 1
        imax = np.searchsorted(ws, [wr_max])[0]
        jmin = np.searchsorted(wis, [wi_min], side='right')[0] - 1
        if imin == -1 or imax == len(self.ws) or jmin == -1:
            raise ValueError(
                "exceed data bounds: imin={} imax={} jmin={} len(ws)={}".format(
                    imin, imax, jmin, len(self.ws)))
        beta_funcs = {}
        for pol in ['M', 'E']:
            for n in range(num_n):
                for m in range(1, num_m + 1):
                    alpha = (pol, n, m)
                    conv = convs[alpha][:, ::-1]
                    if np.all(conv[imin: imax + 1, jmin:]):
                        data = betas[alpha][:, ::-1]
                        beta_funcs[(alpha, 'real')] = RectBivariateSpline(
                            ws[imin: imax + 1], wis[jmin:],
                            data.real[imin: imax + 1, jmin:],
                            kx=3, ky=3)
                        beta_funcs[(alpha, 'imag')] = RectBivariateSpline(
                            ws[imin: imax + 1], wis[jmin:],
                            data.imag[imin: imax + 1, jmin:],
                            kx=3, ky=3)
        return beta_funcs
