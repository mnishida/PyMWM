from __future__ import annotations

import numpy as np

def coefs_cython(
    hole: object, hs: np.ndarray, w: complex
) -> tuple[np.ndarray, np.ndarray]: ...
def ABY_cython(
    w: complex,
    r: float,
    s_all: np.ndarray,
    n_all: np.ndarray,
    hs: np.ndarray,
    e1: complex,
    e2: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def uvABY_cython(
    w: complex,
    r: float,
    s_all: np.ndarray,
    n_all: np.ndarray,
    hs: np.ndarray,
    e1: complex,
    e2: complex,
) -> tuple[np.ndaray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def eig_eq_cython(
    h2: complex, w: complex, pol: str, n: int, e1: complex, e2: complex, r: float
) -> complex: ...
def func_cython(
    h2vec: np.ndarray,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    roots: np.ndarray,
) -> tuple[float, float]: ...
