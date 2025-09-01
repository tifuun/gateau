import unittest
import numpy as np
import tempfile
from pathlib import Path
from multiprocessing import Process

import gateau as gt

class TestGPU(unittest.TestCase):
    def test_crash_gateau(self, for_real=False):
        """
        Test that invalid input data crashes with GPUassert: unknown error 

        Okay, this is a weird test.
        Running gateau "properly" requires LARGE files like 3GB minimum
        so here we test it on nonsense data and EXPECT it to fail.
        But the important part is that it crashes because of the invalid
        data,
        and not some other thing like resources not being found.
        """

        # TODO won't actually clean up if exit due to CUDA errror??

        if for_real is False:
            proc = Process(
                target=self.test_crash_gateau,
                kwargs={'for_real': True},
                )
            proc.start()
            proc.join()

            self.assertNotEqual(
                proc.exitcode,
                35,
                "CUDA driver error! Check your hardware/driver setup "
                "and make sure GPU is forwarded into container!! "
                "This part should crash with a GPUAssert, not "
                "driver error!! "
                )

            self.assertEqual(
                proc.exitcode,
                231,
                "Process crashed, but not because of GPUAssert!! "
                "Check your setup!! "
                )

            print(
                "If the text above says "
                "something about GPUassert "
                "DO NOT WORRY, this is what this test is designed "
                "to cause, it means everything went ok"
                )

            # TODO ultimately we want to capture stderr
            # and check that the C++ code writes 
            # `GPUassert: unknown error`
            # but I do not know how to capture stderr of non-python
            # code.
            # Can try using `subprocess` module
            # instead of `multiprocessing`
            # but that would mean figuring out what interpreter to run
            # with what path, etc.....

            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

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

