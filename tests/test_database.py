#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from pymwm.waveguide import Database


class Load(unittest.TestCase):
    def setUp(self):
        key = dict(
            (
                ("shape", "cylinder"),
                ("size", 0.5),
                ("size2", 0.0),
                ("core", {"RI", 1.0}),
                ("clad", {"book": "Au", "page": "Stewart-DLF"}),
                ("wl_max", 1.0),
                ("wl_min", 0.4),
                ("wl_imag", 5.0),
                ("dw", 1 / 64),
                ("num_n", 6),
                ("num_m", 7),
                ("num_all", 30),
                ("im_factor", 1.0),
            )
        )
        self.database = Database(key)

    def test_load(self):
        with self.assertRaises(IndexError):
            self.database.load()


if __name__ == "__main__":
    unittest.main()
