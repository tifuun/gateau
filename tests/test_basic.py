import unittest
import numpy as np

import gateau as gt

class TestBasic(unittest.TestCase):
    def test_load_lib(self):
        """
        Test that libgateau.so can be loaded correctly.
        """
        self.assertTrue(gt.bindings.load_gateaulib() is not None)

    def test_load_resources_py(self):
        """
        Test that `resources` folder is located correctly by Python code.

        Doesn't test the C++ logic.
        """
        self.assertTrue(
            gt.utilities.get_eta_atm(
                np.array([10e6, 50e6, 100e6, 500e6, 1000e6]),
                0.3,
                45
                )
            is not None
            )

if __name__ == '__main__':
    unittest.main()

