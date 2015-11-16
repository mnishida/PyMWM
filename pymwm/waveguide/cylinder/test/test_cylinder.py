#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nose.tools import assert_equal, assert_true
import numpy as np
import numpy.testing as npt
from scipy import constants


def test_attributes():
    from pymwm.waveguide import create
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'frmin': 1.0, 'frmax': 6.0, 'fimin': -0.6,
                        'nwr': 512, 'nwi': 64, 'num_n': 6, 'num_m': 2}}
    wg = create(params)
    p = params['modes']
    ws = np.linspace(p['wi'], p['wf'], p['nwr'] + 1)
    npt.assert_equal(wg.ws, ws)
    npt.assert_equal(wg.wis, np.linspace(0.0, p['wif'], p['nwi'] + 1))
    assert_equal(wg.r, params['core']['size'] / params['unit'])
    w = wg.ws[0]
    assert_equal(wg.fill(w), 1.0)
    drude_lorentz = -519.83137344754186+91.044638054269981j
    npt.assert_almost_equal(wg.clad(w), drude_lorentz, decimal=10)


def test_beta_pec():
    from pymwm.waveguide import create
    unit = 0.5
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl', 'unit': unit},
              'modes': {'frmin': 1.0, 'frmax': 6.0, 'fimin': -0.6,
                        'nwr': 512, 'nwi': 64, 'num_n': 6, 'num_m': 2}}
    wg = create(params)
    w = wg.ws[0]
    pec0 = [7.95346602j, 18.37306674j, 28.82842093j,
            12.733146j, 23.36389815j]
    npt.assert_almost_equal(wg.beta_pec(w, 0), pec0, decimal=8)
    pec1 = [12.733146j, 23.36389815j, 33.89681301j,
            6.05526191j, 17.74331863j]
    npt.assert_almost_equal(wg.beta_pec(w, 1), pec1, decimal=8)
    pec2 = [17.08950829j, 28.03965425j, 38.7198928j,
            10.13155862j, 22.33139853j]
    npt.assert_almost_equal(wg.beta_pec(w, 2), pec2, decimal=8)
    pec3 = [21.24368294j, 32.52137289j, 43.37247589j,
            13.96821333j, 26.69873441j]
    npt.assert_almost_equal(wg.beta_pec(w, 3), pec3, decimal=8)
    pec4 = [25.27469989j, 36.86880585j, 47.89801786j,
            17.69694614j, 30.92515711j]
    npt.assert_almost_equal(wg.beta_pec(w, 4), pec4, decimal=8)
    pec5 = [29.22117351j, 41.11652189j, 52.32435869j,
            21.36199467j, 35.05194127j]
    npt.assert_almost_equal(wg.beta_pec(w, 5), pec5, decimal=8)


def test_beta_low_freq():
    from pymwm.waveguide import create
    params = {'core': {'shape': 'cylinder', 'size': 0.15,
                       'fill': {'model': 'air'}},
              'clad': {'model': 'gold_dl'},
              'modes': {'frmin': 1.0, 'frmax': 6.0, 'fimin': -0.6,
                        'nwr': 512, 'nwi': 64, 'num_n': 6, 'num_m': 2}}
    wg = create(params)
    betas0 = [1.09741972e-03+7.93750204j, -3.54745588e-04+18.36865209j,
              -3.76568683e-03+28.82754838j, 1.35859873e-01+11.05785572j,
              3.76157700e-01+19.92299197j]
    npt.assert_almost_equal(wg.beta_fi(0)[0], betas0, decimal=8)
    betas1 = [0.00032276+12.72444049j, -0.00185970+23.36244069j,
              -0.00462024+33.8959835j, 0.06127891+5.25491869j,
              0.21223401+15.32923303j]
    npt.assert_almost_equal(wg.beta_fi(1)[0], betas1, decimal=8)
    betas2 = [-2.65502309e-04+17.08456457j, -3.84173718e-03+28.03887897j,
              -5.13409678e-03+38.71905647j, 1.01873719e-01+8.81475775j,
              3.17036453e-01+19.1490023j]
    npt.assert_almost_equal(wg.beta_fi(2)[0], betas2, decimal=8)
    betas3 = [-0.00108714+21.24167529j, -0.00484030+32.52064405j,
              -0.00549378+43.37162207j, 0.14293958+12.15137052j,
              0.52240253+22.64152401j]
    npt.assert_almost_equal(wg.beta_fi(3)[0], betas3, decimal=8)
    betas4 = [-0.00307876+25.27514674j, -0.00545822+36.86806073j,
              -0.00576896+47.89714399j, 0.18667086+15.38026987j,
              1.02744634+26.00286908j]
    npt.assert_almost_equal(wg.beta_fi(4)[0], betas4, decimal=8)
    betas5 = [-0.00543388+29.22197035j, -0.00589659+41.11574901j,
              -0.00599215+52.32346453j, 0.23568143+18.53677139j]
    npt.assert_almost_equal(wg.beta_fi(5)[0][:4], betas5, decimal=8)
