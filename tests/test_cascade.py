import unittest
import numpy as np
from shutil import rmtree

from gateau import cascade

TESTFOLDER  = "test"
ETA_TEST    = 0.9
T_TEST      = 300
NEFF_TEST   = 11
THICKNESS   = 8e-3
TDELTA_TEST = 1e-4

ARR_F       = np.linspace(10, 1000) * 1e9
ARR_ETA     = np.linspace(0.9, 0.5)

REFL_STAGE_AMB  = {"eta_coup"    : ETA_TEST,
                   "T_parasitic" : T_TEST,
                   "group"       : "amb"}

OHMIC_STAGE     = {"eta_coup"    : "Ohmic-Al",
                   "T_parasitic" : T_TEST}

SPILL_ATM_STAGE = {"eta_coup"    : ETA_TEST,
                   "T_parasitic" : "atmosphere"}

REFL_STAGE_ARR  = {"eta_coup"    : (ARR_F, ARR_ETA),
                   "T_parasitic" : T_TEST}

WINDOW_STAGE_CRYO = {"thickness" : THICKNESS, 
                     "tandelta"  : TDELTA_TEST, 
                     "neff"      : NEFF_TEST,
                     "AR"        : False,
                     "T_parasitic_refl" : T_TEST,
                     "T_parasitic_refr" : T_TEST,
                     "cryo_window_flag" : True}

TESTCASCADE = [
        REFL_STAGE_AMB,
        REFL_STAGE_AMB,
        OHMIC_STAGE,
        SPILL_ATM_STAGE,
        REFL_STAGE_ARR,
        WINDOW_STAGE_CRYO,
        REFL_STAGE_AMB
        ]

class TestCascade(unittest.TestCase):
    def test_yaml_write_read(self):
        cascade.save_cascade(TESTCASCADE,
                             TESTFOLDER)        

        test_list = cascade.read_from_folder(TESTFOLDER)

        for stage_ori, stage_load in zip(TESTCASCADE, test_list):
            for key_ori, item_ori in stage_ori.items():
                for key_load, item_load in stage_load.items():
                    if key_ori == key_load:
                        if isinstance(item_ori, tuple):
                            self.assertTrue(np.allclose(item_ori[0],
                                                        item_load[0]))

                            self.assertTrue(np.allclose(item_ori[1],
                                                        item_load[1]))
                        else:
                            self.assertAlmostEqual(item_ori, 
                                                   item_load)

        rmtree(TESTFOLDER)        

if __name__ == "__main__":
    import nose2
    nose2.main()
