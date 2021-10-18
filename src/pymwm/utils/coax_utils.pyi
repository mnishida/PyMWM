from __future__ import annotations

import numpy as np

def eig_eq_with_jac(
    h2: np.ndarray,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    ri: float,
    roots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
def eig_eq_for_min(
    h2: np.ndarray,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    ri: float,
    roots: np.ndarray,
) -> tuple[float, np.ndarray]: ...
def eig_eq(
    h2: np.ndarray,
    w: complex,
    pol: str,
    n: int,
    e1: complex,
    e2: complex,
    r: float,
    ri: float,
    roots: np.ndarray,
) -> np.ndarray: ...

# def func_cython(
#     h2vec: np.ndarray,
#     w: complex,
#     pol: str,
#     n: int,
#     e1: complex,
#     e2: complex,
#     r: float,
#     ri: float,
#     roots: np.ndarray,
# ) -> tuple[float, float]: ...