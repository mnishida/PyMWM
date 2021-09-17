# PyMWM

[![PyPI version][pypi-image]][pypi-link]
[![Anaconda Version][anaconda-v-image]][anaconda-v-link]
[![Lint and Test][github-workflow-image]][github-workflow-link]

[pypi-image]: https://badge.fury.io/py/pymwm.svg
[pypi-link]: https://pypi.org/project/pymwm
[anaconda-v-image]: https://anaconda.org/mnishida/pymwm/badges/version.svg
[anaconda-v-link]: https://anaconda.org/mnishida/pymwm
[github-workflow-image]: https://github.com/mnishida/PyMWM/actions/workflows/pythonapp.yml/badge.svg
[github-workflow-link]: https://github.com/mnishida/PyMWM/actions/workflows/pythonapp.yml

PyMWM is a metallic waveguide mode solver written in Python.

It provides the dispersion relation, i.e. the relation between propagation constant &beta; = &alpha; + i&gamma; (with phase constant &alpha; and attenuation constant &gamma;) and angular frequency &omega;, for cylindrical waveguide and planer waveguide (slits). It also provides the distribution of mode fields. Codes for coaxial waveguides are under development.

## Version
0.2.5

## Install
#### Install and update using pip
```
$ pip install -U pymwm
```
#### Install using conda
```
$ conda install -c mnishida pymwm
```

## Dependencies
- python 3
- numpy
- scipy
- pandas
- pytables
- ray
- matplotlib
- riip
## Uninstall
```
$ pip uninstall pymwm
```
or
```
$ conda deactivate
$ conda remove -n pymwm --all
```

## Usage
Let's consider a cylindrical waveguide whose radius is 0.15&mu;m filled with water (refractive index : 1.333) surrounded by gold.
You can specify the materials by the parameters for [RII_Pandas](https://github.com/mnishida/RII_Pandas).
Wavelength range is set by the parameters 'wl_min' (which is set 0.5 &mu;m here) and 'wl_max' (1.0 &mu;m).
PyMWM compute the dispersion relation the two complex values, &omega; (complex angular frequency) and &beta; (propagation constant).
The range of the imaginary part of &omega; is set from -2&pi;/wl_imag to 0 with the parameter 'wl_imag'.
Usually, the cylindrical waveguide mode is specified by two integers, n and m.
The number of sets indexed by n and m are indicated by the parameters 'num_n' and 'num_m', respectively.
```
>>> import pymwm
>>> params = {
     'core': {'shape': 'cylinder', 'size': 0.15, 'fill': {'RI': 1.333}},
     'clad': {'book': 'Au', 'page': 'Stewart-DLF'},
     'bounds': {'wl_max': 1.0, 'wl_min': 0.5, 'wl_imag': 10.0},
     'modes': {'num_n': 6, 'num_m': 2}
     }
>>> wg = pymwm.create(params)
```
If the parameters are set for the first time, the creation of waveguide-mode object will take a quite long time, because a sampling survey of &beta;s in the complex plane of &omega; will be conducted and the obtained data is registered in the database.
The second and subsequent creations are done instantly.
You can check the obtained waveguide modes in the specified range by showing the attribute 'modes';
```
>>> wg.modes
{'h': [('E', 1, 1),
  ('E', 2, 1),
  ('E', 3, 1),
  ('E', 4, 1),
  ('M', 0, 1),
  ('M', 1, 1)],
 'v': [('E', 0, 1),
  ('E', 1, 1),
  ('E', 2, 1),
  ('E', 3, 1),
  ('E', 4, 1),
  ('M', 1, 1)]}
```
where 'h' ('v') means that the modes have horizontally (vertically) oriented electric fields on the x axis. The tuple (pol, n, m) with pol being 'E' or 'M' indicates TE-like or TM-like mode indexed by n and m.
You can get &beta; at &omega;=8.0 rad/&mu;m for TM-like mode with n=0 and m=1 by
```
>>> wg.beta(8.0, ('M', 0, 1))
(0.06187318716518497+10.363105296313996j)
```
and for TE-like mode with n=1 and m=2 by
```
>>> wg.beta(8.0, ('E', 1, 2))
(0.14261514314942403+19.094726281995463j)
```
For more information, see the [tutorial notebook](https://github.com/mnishida/PyMWM/blob/master/docs/notebooks/00_tutorial.ipynb) and [User's Guide](https://pymwm.readthedocs.io/en/latest/).

## Examples
### Cylindrical waveguide
#### Propagation constants
<img src="https://github.com/mnishida/PyMWM/wiki/images/phase_constant.png"
     alt="phase constant" title="phase constant" width="400"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/attenuation_constant.png"
     alt="attenuation constant" title="attenuation constant" width="400"/>

#### Electric field and magnetic field distributions
* TE01
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE01_electric.png"
     alt="TE01 electric field" title="TE01 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE01_magnetic.png"
     alt="TE01 magnetic field" title="TE01 magnetic field" width="300"/>
* HE11
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE11_electric.png"
     alt="HE11 electric field" title="HE11 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE11_magnetic.png"
     alt="HE11 magnetic field" title="HE11 magnetic field" width="300"/>
* HE12
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE12_electric.png"
     alt="HE12 electric field" title="HE12 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE12_magnetic.png"
     alt="HE12 magnetic field" title="HE12 magnetic field" width="300"/>
* HE21
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE21_electric.png"
     alt="HE21 electric field" title="HE21 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE21_magnetic.png"
     alt="HE21 magnetic field" title="HE21 magnetic field" width="300"/>
* HE31
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE31_electric.png"
     alt="HE31 electric field" title="HE31 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/HE31_magnetic.png"
     alt="HE31 magnetic field" title="HE31 magnetic field" width="300"/>
* TM01
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM01_electric.png"
     alt="TM01 electric field" title="TM01 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM01_magnetic.png"
     alt="TM01 magnetic field" title="TM01 magnetic field" width="300"/>
* TM02
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM02_electric.png"
     alt="TM02 electric field" title="TM02 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM02_magnetic.png"
     alt="TM02 magnetic field" title="TM02 magnetic field" width="300"/>
* EH11
<img src="https://github.com/mnishida/PyMWM/wiki/images/EH11_electric.png"
     alt="EH11 electric field" title="EH11 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/EH11_magnetic.png"
     alt="EH11 magnetic field" title="EH11 magnetic field" width="300"/>
* EH21
<img src="https://github.com/mnishida/PyMWM/wiki/images/EH21_electric.png"
     alt="EH21 electric field" title="EH21 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/EH21_magnetic.png"
     alt="EH21 magnetic field" title="EH21 magnetic field" width="300"/>

### Slit waveguide
#### Propagation constants
<img src="https://github.com/mnishida/PyMWM/wiki/images/phase_constant_slit.png"
     alt="phase constant (slit)" title="phase constant (slit)" width="400"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/attenuation_constant_slit.png"
     alt="attenuation constant (slit)" title="attenuation constant (slit)" width="400"/>

#### Electric field and magnetic field distributions
* TE1
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE1_electric.png"
     alt="TE1 electric field" title="TE1 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE1_magnetic.png"
     alt="TE1 magnetic field" title="TE1 magnetic field" width="300"/>
* TE2
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE2_electric.png"
     alt="TE2 electric field" title="TE2 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE2_magnetic.png"
     alt="TE2 magnetic field" title="TE2 magnetic field" width="300"/>
* TE3
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE3_electric.png"
     alt="TE3 electric field" title="TE3 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE3_magnetic.png"
     alt="TE3 magnetic field" title="TE3 magnetic field" width="300"/>
* TE4
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE4_electric.png"
     alt="TE4 electric field" title="TE4 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TE4_magnetic.png"
     alt="TE4 magnetic field" title="TE4 magnetic field" width="300"/>
* TM0
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM0_electric.png"
     alt="TM0 electric field" title="TM0 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM0_magnetic.png"
     alt="TM0 magnetic field" title="TM0 magnetic field" width="300"/>
* TM1
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM1_electric.png"
     alt="TM1 electric field" title="TM1 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM1_magnetic.png"
     alt="TM1 magnetic field" title="TM1 magnetic field" width="300"/>
* TM2
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM2_electric.png"
     alt="TM2 electric field" title="TM2 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM2_magnetic.png"
     alt="TM2 magnetic field" title="TM2 magnetic field" width="300"/>
* TM3
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM3_electric.png"
     alt="TM3 electric field" title="TM3 electric field" width="300"/>
<img src="https://github.com/mnishida/PyMWM/wiki/images/TM3_magnetic.png"
     alt="TM3 magnetic field" title="TM3 magnetic field" width="300"/>
