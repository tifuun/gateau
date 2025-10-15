import unittest
import numpy as np

import gateau as gt

class TestTyping(unittest.TestCase):
    def test_typing_simulator(self):
        s = gt.simulator.simulator().initialise(
            0.0,
            0.0,
            0.0,
            gt.scan_patterns.daisy,
            {'f0_ch': 0, 'f_sample': 0},
            {'eta_ap': 0, },
            {'T_atm': 0, 'path': 0, 'dx': 0, 'dy': 0, 'h_column': 0, 'v_wind': 0},
            {'I_nu': 0, 'az_src': 0, 'el_src': 0, 'f_src': 0, },
            [],
            )

if __name__ == '__main__':
    unittest.main()

