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

            #gt.bindings.run_gateau(
            #    instrument = {
            #        'filterbank': np.array([]),
            #        'pointings': np.array([0, 1]),
            #        'nf_ch': 0,
            #        'f_sample': 0,
            #        'delta': 0,
            #        'eta_pb': 0,
            #        },
            #    telescope = {
            #        'az_scan': np.array([]),
            #        'eta_ap': np.array([]),
            #        'el_scan': np.array([]),
            #        },
            #    atmosphere = {
            #        'T_atm': 0,
            #        'v_wind': 0,
            #        'h_column': 0,
            #        'dx': 0,
            #        'dy': 0,
            #        'path': tmpdir,
            #        },
            #    source = {
            #        'I_nu': np.array([]),
            #        'az_scan': np.array([]),
            #        'az_src': np.array([0, 1]),
            #        'el_src': np.array([0, 1]),
            #        'f_src': np.array([0, 1]),
            #        },
            #    cascade = {
            #        'eta_stage': np.array([0, 1]),
            #        'psd_stage': np.array([0, 1]),
            #        'num_stage': 0,
            #        },
            #    nTimes = 0,
            #    outpath = '/tmp',
            #    )
            gt.bindings.run_gateau(
                instrument = {
                    'filterbank': np.linspace(90e9, 110e9, 100),  # 100 frequency channels from 90 GHz to 110 GHz
                    'pointings': np.array([[180.0, 45.0], [181.0, 46.0]]),  # 2 pointings: azimuth, elevation in degrees
                    'nf_ch': 100,  # Number of frequency channels
                    'f_sample': 10.0,  # 10 Hz sample rate
                    'delta': 0.01,  # Beam width in radians?
                    'eta_pb': 0.8,  # Peak efficiency of the primary beam
                },
                telescope = {
                    'az_scan': np.linspace(180.0, 182.0, 100),  # Azimuth scan from 180° to 182°
                    'eta_ap': np.full(100, 0.7),  # Aperture efficiency (example: 70%)
                    'el_scan': np.full(100, 45.0),  # Constant elevation scan at 45°
                },
                atmosphere = {
                    'T_atm': 270.0,  # Atmospheric temperature in Kelvin
                    'v_wind': 10.0,  # Wind speed in m/s
                    'h_column': 1.0,  # Precipitable water vapor column in mm
                    'dx': 100.0,  # Spatial resolution in x-direction in meters
                    'dy': 100.0,  # Spatial resolution in y-direction in meters
                    'path': tmpdir,  # Temporary directory for intermediate files
                },
                source = {
                    'I_nu': np.ones(100) * 1e-20,  # Flat spectrum source intensity (in arbitrary units)
                    'az_scan': np.linspace(180.0, 182.0, 100),  # Matching the telescope az scan
                    'az_src': np.array([180.5]),  # Source at az = 180.5°
                    'el_src': np.array([45.0]),  # Source at el = 45.0°
                    'f_src': np.linspace(90e9, 110e9, 100),  # Source frequencies match instrument
                },
                cascade = {
                    'eta_stage': np.array([0.9, 0.8, 0.85]),  # Efficiencies at each stage
                    'psd_stage': np.array([1e-16, 1e-17, 1e-18]),  # Power spectral densities per stage
                    'num_stage': 3,  # Number of cascade stages
                },
                nTimes = 100,  # Number of time samples
                outpath = '/tmp',  # Output directory
            )

if __name__ == '__main__':
    unittest.main()

