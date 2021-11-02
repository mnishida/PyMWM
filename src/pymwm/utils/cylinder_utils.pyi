from __future__ import annotations

import numpy as np

# def uvABY_cython(
#     w: complex,
#     r: float,
#     s_all: np.ndarray,
#     n_all: np.ndarray,
#     m_all: np.ndarray,
#     hs: np.ndarray,
#     e1: complex,
#     e2: complex,
#     u_pec: np.ndarray,
#     jnu_pec: np.ndarray,
#     jnpy_pec: np.ndarray,
# ) -> tuple[np.ndaray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def props_cython(
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
) -> tuple[
    np.ndaray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]: ...
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
def eig_eq_cython(
    h2vec: np.ndarray,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    roots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
