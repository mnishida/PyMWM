from __future__ import annotations

def f_fp_fpp_cython(
    x: float, r_ratio: float, n: int, pol: str
) -> tuple[float, float, float]: ...
def f_fprime_cython(
    x: float, r_ratio: float, n: int, pol: str
) -> tuple[float, float]: ...
def f_cython(x: float, r_ratio: float, n: int, pol: str) -> float: ...
def fprime_cython(x: float, r_ratio: float, n: int, pol: str) -> float: ...
