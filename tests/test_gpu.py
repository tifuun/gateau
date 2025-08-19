import unittest
import numpy as np
import tempfile
from pathlib import Path

import gateau as gt

class TestGPU(unittest.TestCase):
    def test_run_gateau(self):
        """
        Test basic invocation of gateau

        mayybe not valid parameters now arend plz giv minimum example
        """

        # TODO won't actually clean up if exit due to CUDA errror??
        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            file1 = tmpdir / 'foo.datp'
            (tmpdir / 'prepd').mkdir()
            (tmpdir / 'prepd' / 'atm_meta.datp').write_text('0')

            gt.bindings.run_gateau(
                instrument = {
                    'filterbank': np.array([]),
                    'pointings': np.array([0, 1]),
                    'nf_ch': 0,
                    'f_sample': 0,
                    'delta': 0,
                    'eta_pb': 0,
                    },
                telescope = {
                    'az_scan': np.array([]),
                    'eta_ap': np.array([]),
                    'el_scan': np.array([]),
                    },
                atmosphere = {
                    'T_atm': 0,
                    'v_wind': 0,
                    'h_column': 0,
                    'dx': 0,
                    'dy': 0,
                    'path': tmpdir,
                    },
                source = {
                    'I_nu': np.array([]),
                    'az_scan': np.array([]),
                    'az_src': np.array([0, 1]),
                    'el_src': np.array([0, 1]),
                    'f_src': np.array([0, 1]),
                    },
                cascade = {
                    'eta_stage': np.array([0, 1]),
                    'psd_stage': np.array([0, 1]),
                    'num_stage': 0,
                    },
                nTimes = 0,
                outpath = '/tmp',
                )

if __name__ == '__main__':
    unittest.main()

