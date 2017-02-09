#!/usr/bin/env python
import os
from collections import OrderedDict
from typing import Dict
from bsddb3 import dbshelve
import numpy as np
import pandas as pd
from pandas import DataFrame


catalog_columns = OrderedDict((
    ('id', int), ('shape', str), ('size', float), ('core', str), ('clad', str),
    ('lmax', float), ('lmin', float), ('limag', float), ('dw', float),
    ('num_n', int), ('num_m', int), ('im_factor', float), ('EM', str),
    ('n', int), ('m', int)))
data_columns = OrderedDict((
    ('conv', bool), ('beta_real', float), ('beta_imag', float)))


def set_columns_dtype(df: DataFrame, columns: Dict):
    """Set data type of each column in the DataFrame."""
    for key, val in columns.items():
        df[key] = df[key].astype(val)


dirname = "/home/mnishida/.pymwm"
files = os.listdir(dirname)
# files = []
idx = 0
catalog = pd.DataFrame(columns=catalog_columns.keys())
store = pd.HDFStore('pymwm_data.h5')
for filename in files:
    print(filename)
    s = dbshelve.open(os.path.join(dirname, filename), 'r')
    params = filename.split('_')
    params = params[:-1] + [params[-1].rstrip('.db')]
    shape = params[0]
    size = float(params[2])
    ind_core = params.index('core')
    ind_clad = params.index('clad')
    ind_end = len(params)
    if ind_clad - ind_core == 2:
        core = params[ind_core + 1]
        if core == 'air':
            core = 'RI_{}'.format(1.0)
        elif core == 'water':
            core = 'RI_{}'.format(1.333)
        elif core == 'BK7':
            core = 'RI_{}'.format(1.51633)
    else:
        if params[ind_core + 1] == 'dielectric':
            eps = complex(params[ind_core + 3])
            core = 'RI_{}'.format(np.sqrt(eps))
        else:
            core = "_".join(params[ind_core + 1: ind_clad])
    if ind_end - ind_clad == 2:
        clad = params[ind_clad + 1]
        if clad == 'air':
            clad = 'RI_{}'.format(1.0)
        elif clad == 'water':
            clad = 'RI_{}'.format(1.333)
        elif clad == 'BK7':
            clad = 'RI_{}'.format(1.51633)
    else:
        if params[ind_clad + 1] == 'dielectric':
            eps = complex(params[ind_clad + 3])
            clad = 'RI_{}'.format(np.sqrt(eps))
        else:
            clad = "_".join(params[ind_clad + 1: ind_end])

    for key, val in s.items():
        txt = key.decode('utf-8')
        if shape == 'slit':
            lmax, lmin, limag, dw, num_n, im_factor = txt.split('_')
            num_m = '1'
        else:
            lmax, lmin, limag, dw, num_n, num_m, im_factor = txt.split('_')
        lmax = float(lmax)
        lmin = float(lmin)
        limag = float(limag)
        dw = float(dw)
        num_m = int(num_m)
        num_n = int(num_n)
        im_factor = float(im_factor)
        ind_wmin = int(np.floor(2 * np.pi / lmax / dw))
        ind_wmax = int(np.ceil(2 * np.pi / lmin / dw))
        ind_wimag = int(np.ceil(2 * np.pi / limag / dw))
        ws = np.arange(ind_wmin, ind_wmax + 1) * dw
        wis = -np.arange(ind_wimag + 1) * dw
        convs = val['convs']
        betas = val['betas']
        for EM, n, m in sorted(convs.keys()):
            se = pd.Series([idx, shape, size, core, clad, lmax, lmin, limag,
                            dw, num_n, num_m, im_factor,
                            EM, n, m], index=catalog_columns.keys())
            catalog = catalog.append(se, ignore_index=True)
            conv = convs[(EM, n, m)].ravel()
            beta = betas[(EM, n, m)].ravel()
            df = pd.DataFrame(
                {'conv': conv, 'beta_real': beta.real, 'beta_imag': beta.imag},
                columns=data_columns.keys())
            set_columns_dtype(df, data_columns)
            store.append('id_{}'.format(idx), df)
            idx += 1
set_columns_dtype(catalog, catalog_columns)
catalog.set_index('id', inplace=True)
store['catalog'] = catalog
store.close()
os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 " +
          "--complib=blosc pymwm_data.h5 pymwm_data_comp.h5")
os.system("mv pymwm_data_comp.h5 pymwm_data.h5")
