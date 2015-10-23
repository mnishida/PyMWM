# -*- coding: utf-8 -*-
import numpy as np


class Muller():
    """A class defining a nonlinear solver based on Muller's method.

    Muller method:
        D.E. Muller, "A method for solving algebraic equation using an
        automatic computer", Mathematical Tables and other Aids to
        Computation, No. 10, 208 (1956).
        P. Henrici, "Elements of numerical analysis" (Wiley, N.Y., 1964).
     Attributes:
        func: A complex function whose root is desired
        num_roots: An integer indicating the numbers of roots to be found.
        xtol: A float >= 0. The routine converges when a root is known to lie
            within xtol of the value return. (defulat 1e-5)
        rtol: A float >= 0. The routine converges when a root is known to lie
            within rtol times the value returned. (defulat 1e-5)
        ftol: A float >= 0 indicating the absolute tolerance for the residual.
        maxiter: An integer >=0 0. If convergence is not achieved in maxiter
            iterations, an error is raised. (defulat 1000)
        roots: A list of complexes indicating the found roots of func.
    """

    def __init__(self, func, num_roots, xtol=1.0e-5, rtol=1.0e-5,
                 ftol=1.0e-5, maxiter=1000):
        """Init Muller class.

        Args:
            func: A complex function whose root is desired
            num_roots: An integer indicating the numbers of roots to be found.
            xtol: A float >= 0. The routine converges when a root is known to
                lie within xtol of the value return. (defulat 1e-5)
            rtol: A float >= 0. The routine converges when a root is known to
                lie within rtol times the value returned. (defulat 1e-5)
            ftol: A float >= 0 indicating the absolute tolerance for the
                residual. (defulat 1e-5)
            maxiter: An integer >=0 0. If convergence is not achieved in
                maxiter iterations, an error is raised. (defulat 1000)
        """
        self.func = func
        self.roots = []
        self.num_roots = num_roots
        self.xtol = xtol
        self.rtol = rtol
        self.ftol = ftol
        self.maxiter = maxiter
        self.iter = []

    def __call__(self, xi=0.0):
        """Find roots.

        Args:
            xi: A complex indicating the initial approximation for the root.
                (default 0.0)
        """
        xinit = xi
        for j in range(self.num_roots):
            xs = np.array([xinit - 1.0, xinit + 1.0, xinit], dtype=complex)
            fs = np.array([self.devided_func(x) for x in xs],
                          dtype=complex)
            h = xs[2] - xs[1]
            q = h / (xs[1] - xs[0])
            for i in range(self.maxiter):
                A = q * fs[2] - q * (1 + q) * fs[1] + q ** 2 * fs[0]
                B = (2 * q + 1) * fs[2] - (1 + q) ** 2 * fs[1] + q ** 2 * fs[0]
                C = (1 + q) * fs[2]
                Q = np.sqrt(B ** 2 - 4 * A * C)
                D = B + Q
                E = B - Q
                if abs(D) < abs(E):
                    D = E
                xs[0] = xs[1]
                xs[1] = xs[2]
                fs[0] = fs[1]
                fs[1] = fs[2]
                if abs(D) <= self.xtol:
                    h = 0.85
                    xs[2] += h
                else:
                    q = -2.0 * C / D
                    h *= q
                    xs[2] += h
                    absf = 100. * abs(fs[2])
                    while True:
                        fs[2] = self.devided_func(xs[2])
                        if abs(fs[2]) < absf:
                            break
                        h *= 0.5
                        xs[2] -= h
                if (abs(self.func(xs[2])) <= self.ftol or
                    abs(h) <= self.xtol or
                    abs(h / xs[2]) <= self.rtol):
                    break
            self.iter.append(i + 1)
            if i == self.maxiter - 1:
                raise Exception
            self.roots.append(xs[2])
            xinit = xs[2] + 0.85

    def devided_func(self, x):
        """Return the value of func(x) dividded by the product
        (x - roots[0])(x - roots[1])...

        Args:
            x: A complex indicating test root

        Returns:
            y: A complex indicating the devided value of func(x).
        """
        y = self.func(x)
        for x0 in self.roots:
            denom = x - x0
            while (abs(denom) < 1e-8):  # avoid division by a small number
                denom += 1.0e-8
            y = y / denom
        return y
