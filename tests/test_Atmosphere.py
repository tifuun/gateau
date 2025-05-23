"""
@file
Test atmosphere module.
"""

import os
import numpy as np
import unittest
import tiempo2.Atmosphere as TAtm
import resources.InputDicts as TInp
from nose2.tools import params

NXY = 160

class TestAtmosphere(unittest.TestCase):
    @params(1,2,3)
    def test_prepAtmospherePWV(self, num):
        screen, nx, ny = TAtm.prepAtmospherePWV(TInp.AtmDict, TInp.TelDict, number=num)
        self.assertEqual(nx, num * NXY)
        self.assertEqual(ny, NXY)
        self.assertEqual(screen.shape, (num * NXY, NXY))

    @classmethod
    def tearDownClass(self):
        pass

if __name__ == "__main__":
    import nose2
    nose2.main()

