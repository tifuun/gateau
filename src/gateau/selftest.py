import numpy as np
import tempfile
import h5py
from pathlib import Path

from gateau.simulator import simulator
from gateau.scan_patterns import stare

AXSIZE = 3
ANGLIM = 42 / 3600

ETATEST = 0.666
TTEST = 666

source = np.zeros((AXSIZE, AXSIZE, AXSIZE))
source[AXSIZE // 2, AXSIZE // 2, :] = 1

az = np.array([-ANGLIM, 0, ANGLIM])
el = az + 60
f_src = np.array([290, 300, 310]) * 1e9

source_dict = {
        "I_nu"      : source,
        "az_src"    : az,
        "el_src"    : el,
        "f_src"     : f_src,
        }

cascade_list = [{
        "eta_coup"      : ETATEST,
        "T_parasitic"   : TTEST,
        }]

telescope_dict = {
        "eta_taper"        : ETATEST,
        "s_rms"         : 42,
        }

instrument_dict = {
        "f0_ch"         : 300e9,
        "nf_ch"         : 1,
        "R"             : 500,
        "f_sample"      : 158,
        "eta_circuit"   : ETATEST
        }

atmosphere_dict = {
        "T_atm"          : TTEST,
        "path"          : ".",
        "dx"            : 0.2,
        "dy"            : 0.2,
        "h_column"      : 1000,
        "v_wind"        : 0,
        "PWV0"          : 0.5
        }

DATP = (
    "-5.460e-03 -5.460e-03 -5.459e-03\n"
    "-5.459e-03 -5.459e-03 -5.458e-03\n"
    "-5.455e-03 -5.455e-03 -5.454e-03\n"
    "-5.452e-03 -5.451e-03 -5.452e-03\n"
    "-5.460e-03 -5.460e-03 -5.459e-03\n"
    "-5.459e-03 -5.459e-03 -5.458e-03\n"
    "-5.455e-03 -5.455e-03 -5.454e-03\n"
    "-5.452e-03 -5.451e-03 -5.452e-03\n"
    )

ATM_META = "1 8 3 -0.01 0.01"

def selftest():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        (tmpdir / 'prepd').mkdir()
        (tmpdir / 'prepd' / '0.datp').write_text(DATP)
        (tmpdir / 'prepd' / 'atm_meta.datp').write_text(ATM_META)

        atmosphere_dict["path"] = str(tmpdir)
        # TODO very dirty!

        interface = simulator(verbose=True)

        n_hr = 0.42

        return_dict = interface.initialise(
            n_hr * 3600,
            0,
            60,
            stare,
            instrument_dict, 
            telescope_dict,
            atmosphere_dict,
            source_dict,
            cascade_list,
            return_full=True
            )

        seed = 420
        
        outpath_ref = str(tmpdir / 'ref')
        interface.run(outname=outpath_ref, seed=seed)

        outpath = str(tmpdir / 'output')
        interface.run(outname=outpath, seed=seed)

        f = h5py.File(f"{outpath}.h5", 'r')
        f_ref = h5py.File(f"{outpath_ref}.h5", 'r')

        times = f["OBSATTRS"]["times"][:].astype(np.float64)

        freq = f["OBSATTRS"]["frequencies"][:].astype(np.float64)

        tods = f["SPAXEL0"]["data"][:].astype(np.float64)

        ra = f["OBSATTRS"]["az"][:]
        dec = f["OBSATTRS"]["el"][:]

        tods_ref = f_ref["SPAXEL0"]["data"][:].astype(np.float64)

# Check if output length is as specified
        assert(times.size == interface.n_times)
        assert(ra.size == interface.n_times)
        assert(dec.size == interface.n_times)
            
# Check if output frequency array length is as specified
        assert(freq.size == interface.instrument["f_ch_arr"].size)

# Check if output has correct shape
        assert(times.size == tods.shape[0])
        assert(freq.size == tods.shape[1])
            
# Check if generated hdf5 is same as ref
        np.allclose(tods.ravel(), tods_ref.ravel())

