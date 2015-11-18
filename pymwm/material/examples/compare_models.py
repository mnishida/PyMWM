#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pymwm.material import Material
import numpy as np
import matplotlib.pyplot as plt


gold_d = Material({'model': 'gold_d'})
gold_dl = Material({'model': 'gold_dl'})
gold_jc = Material({'model': 'gold_jc'})
# gold_palik = Material({'model': 'gold_palik'})
jc = gold_jc._johnson_christy
jc_ls = 2 * np.pi / jc.ws
# palik = gold_palik._palik
# palik_ls = 2 * np.pi / palik.ws
# plt.plot(palik_ls, gold_palik(palik.ws).real, "o")
plt.plot(jc_ls, np.vectorize(gold_dl)(jc.ws).real, "b-", lw=2.0,
         label='Drude-Lorentz, real')
plt.plot(jc_ls, np.vectorize(gold_dl)(jc.ws).imag, "b--", lw=2.0,
         label='Drude-Lorentz, imag')
plt.plot(jc_ls, np.vectorize(gold_d)(jc.ws).real, "g-", lw=2.0,
         label='Drude, real')
plt.plot(jc_ls, np.vectorize(gold_d)(jc.ws).imag, "g--", lw=2.0,
         label='Drude, imag')
plt.plot(jc_ls, np.vectorize(gold_jc)(jc.ws).real, "ko", ms=7.0,
         label='Johnson-Christy, real')
plt.plot(jc_ls, np.vectorize(gold_jc)(jc.ws).imag, "k^", ms=8.5,
         label='Johnson-Christy, imag')
plt.xlim(0.4, 1.0)
plt.ylim(-50, 10)
plt.tick_params(labelsize=20)
plt.xlabel("Wavelength [$\mu \mathrm{m}$]", size=20)
plt.ylabel("Relative Permittivity", size=20)
plt.tight_layout()
plt.legend(loc=3)
plt.show()
