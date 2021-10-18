import numpy as np
import numpy.testing as npt

from pymwm.coax.samples import Samples

params: dict = {
    "core": {"shape": "coax", "r": 0.15, "ri": 0.1, "fill": {"RI": 1.0}},
    "clad": {"book": "Au", "page": "Stewart-DLF", "bound_check": False},
    "modes": {
        "wl_max": 20.0,
        "wl_min": 1.0,
        "wl_imag": 50.0,
        "dw": 1.0 / 64,
        "num_n": 6,
        "num_m": 2,
    },
}


# def test_attributes():
#     size: float = params["core"]["r"]
#     fill = params["core"]["fill"]
#     clad = params["clad"]
#     p = params["modes"]
#     size2: float = params["core"]["ri"]
#     wg = Samples(size, fill, clad, p, size2)
#     ind_w_min = int(np.floor(2 * np.pi / p["wl_max"] / p["dw"]))
#     ind_w_max = int(np.ceil(2 * np.pi / p["wl_min"] / p["dw"]))
#     ind_w_imag = int(np.ceil(2 * np.pi / p["wl_imag"] / p["dw"]))
#     ws = np.arange(ind_w_min, ind_w_max + 1) * p["dw"]
#     wis = -np.arange(ind_w_imag + 1) * p["dw"]
#     npt.assert_equal(wg.ws, ws)
#     npt.assert_equal(wg.wis, wis)


# def test_beta2_pec(num_regression):
#     size: float = params["core"]["r"]
#     fill = params["core"]["fill"]
#     clad = params["clad"]
#     p = params["modes"]
#     size2: float = params["core"]["ri"]
#     wg = Samples(size, fill, clad, p, size2)
#     w = 2 * np.pi / 5.0
#     num_regression.check({n: wg.beta2_pec(w, n).real for n in range(6)})


# def test_compare_with_cylinder():
#     import pymwm.cylinder

#     size: float = params["core"]["r"]
#     fill = params["core"]["fill"]
#     clad = params["clad"]
#     p = params["modes"]
#     size2 = 1e-5
#     wg1 = Samples(size, fill, clad, p, size2)
#     wg2 = pymwm.cylinder.samples.Samples(size, fill, clad, p)
#     w = 2 * np.pi / 20.0

#     val1 = wg1.beta2_pec(w, 0)
#     val2 = wg2.beta2_pec(w, 0)
#     try:
#         npt.assert_allclose(val1[1:3], val2[:2], rtol=0.1)
#     except Exception as e:
#         print(f"n={0}")
#         print(val1[1:3])
#         print(val2[:2])
#         raise e
#     try:
#         npt.assert_allclose(val1[3:], val2[3:], rtol=0.1)
#     except Exception as e:
#         print(f"n={0}")
#         print(val1[3:])
#         print(val2[3:])
#         raise e
#     val1, success1 = wg1.beta2_w_min(0)
#     val2, success2 = wg2.beta2_w_min(0)
#     try:
#         npt.assert_array_equal(success1[1:3], success2[:2])
#         npt.assert_allclose(val1[1:3], val2[:2], rtol=0.001, atol=0.001)
#     except Exception as e:
#         print(f"n={0}")
#         print("coax", success1[1:3], val1[1:3])
#         print("cylinder", success2[:2], val2[:2])
#         raise e
#     try:
#         npt.assert_array_equal(success1[3:], success2[3:])
#         npt.assert_allclose(val1[3:], val2[3:], rtol=0.001)
#     except Exception as e:
#         print(f"n={0}")
#         print("coax", success1[3:], val1[3:])
#         print("cylinder", success2[3:], val2[3:])
#         raise e
#     for n in range(1, 6):
#         try:
#             val1 = wg1.beta2_pec(w, n)
#             val2 = wg2.beta2_pec(w, n)
#             npt.assert_allclose(val1, val2, rtol=0.1)
#         except Exception as e:
#             print(f"n={n}")
#             print(val1)
#             print(val2)
#             raise e
#         try:
#             val1 = wg1.beta2_w_min(n)
#             val2 = wg2.beta2_w_min(n)
#             npt.assert_allclose(val1, val2, rtol=0.001)
#         except Exception as e:
#             print(f"n={n}")
#             print("coax", val1)
#             print("cylinder", val2)
#             raise e


def test_beta2_w_min(num_regression):
    size: float = params["core"]["r"]
    fill = params["core"]["fill"]
    clad = params["clad"]
    p = params["modes"]
    size2: float = params["core"]["ri"]
    wg = Samples(size, fill, clad, p, size2)
    d = {}
    for n in range(5):
        val = wg.beta2_w_min(n)
        d[f"{n}_real"] = val[0].real
        d[f"{n}_imag"] = val[0].imag
        d[f"{n}_success"] = val[1]
    num_regression.check(d)
