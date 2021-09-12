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
    m_all: np.ndarray,
    hs: np.ndarray,
    e1: complex,
    e2: complex,
    u_pec: np.ndarray,
    jnu_pec: np.ndarray,
    jnpu_pec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def uvABY_cython(
    w: complex,
    r: float,
    s_all: np.ndarray,
    n_all: np.ndarray,
    m_all: np.ndarray,
    hs: np.ndarray,
    e1: complex,
    e2: complex,
    u_pec: np.ndarray,
    jnu_pec: np.ndarray,
    jnpy_pec: np.ndarray,
) -> tuple[np.ndaray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
