"""
@file
Test atmosphere module.
"""

import os
import numpy as np
import unittest
import tiempo2.Interface as TInterf
import resources.InputDicts as TInp
import tiempo2.BindCPU as tbcpu
from nose2.tools import params

import matplotlib.pyplot as pt

class TestAtmosphere(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interf = TInterf.Interface(verbose=False)
        cls.interf.setInstrumentDict(TInp.InstDict)
        cls.interf.setAtmosphereDict(TInp.AtmDict)
        cls.interf.setTelescopeDict(TInp.TelDict)
        cls.interf.setSourceDict(TInp.SourceDict)

        cls.interf.initSetup()
        cls.interf.initSource()

        cls.PWV = 1
        cls.nPWV = 100

        cls.nbins_add = 34

        cls.GPU_flag = True
        try:
            lib = tbcpu.loadTiEMPO2lib_CUDA()
        except OSError:
            cls.GPU_flag = False
    
    def test_getEtaAtm(self):
        res, f = self.interf.getEtaAtm(self.PWV)
        self.assertEqual(res.size, self.interf.instrumentDict["nf_src"])
        self.assertEqual(f.size, self.interf.instrumentDict["nf_src"])
        self.assertTrue(np.array_equal(f, self.interf.instrumentDict["f_src"]))

    def test_getNEP(self):
        res, f = self.interf.getNEP(self.PWV)
        self.assertEqual(res.size, self.interf.instrumentDict["nf_ch"])
        self.assertEqual(f.size, self.interf.instrumentDict["nf_ch"])
        self.assertTrue(np.array_equal(f, self.interf.instrumentDict["f_ch_arr"]))
    
    @params(True, False)
    def test_getSourceSignal(self, chop):
        res, f = self.interf.getSourceSignal(0, 0, self.PWV, chop)
        self.assertEqual(res.size, self.interf.instrumentDict["nf_ch"])
        self.assertEqual(f.size, self.interf.instrumentDict["nf_ch"])
        self.assertTrue(np.array_equal(f, self.interf.instrumentDict["f_ch_arr"]))
    
    @params("CPU", "GPU")
    def test_runSimulation(self, dev):
        if dev == "GPU" and not self.GPU_flag:
            print("*No GPU library found, skipping GPU tests*")
            return

        res, t, f = self.interf.runSimulation(1, device=dev, verbosity=0)

        shape_out = (t.size, self.interf.instrumentDict["nf_ch"])

        self.assertEqual(res["signal"].shape, shape_out)
        self.assertEqual(f.size, self.interf.instrumentDict["nf_ch"])
        self.assertEqual(t.size, self.interf.instrumentDict["f_sample"])
        self.assertTrue(np.array_equal(f, self.interf.instrumentDict["f_ch_arr"]))

    def test_calcW2K(self):
        res, t, f = self.interf.runSimulation(1, verbosity=0)
        w2k, f = self.interf.calcW2K(self.nPWV, verbosity=0)
        
        shape_out = (t.size, self.interf.instrumentDict["nf_ch"])
        shape_w2k = (self.nPWV, self.interf.instrumentDict["nf_ch"])

        self.assertTrue(np.array_equal(f, self.interf.instrumentDict["f_ch_arr"]))
        self.assertEqual(w2k["power"].shape, shape_w2k)
        self.assertEqual(w2k["temperature"].shape, shape_w2k)

        self.assertEqual(w2k["a"].size, self.interf.instrumentDict["nf_ch"])
        self.assertEqual(w2k["b"].size, self.interf.instrumentDict["nf_ch"])
        
        self.interf.Watt2Kelvin(res, w2k)
        self.assertEqual(res["signal"].shape, shape_out)

    def test_calcSignalPSD(self):
        res, t, f = self.interf.runSimulation(1, verbosity=0)

        psd, f_psd = self.interf.calcSignalPSD(res, t)

        shape_psd = (t.size, self.interf.instrumentDict["nf_ch"])
        self.assertEqual(psd.shape, shape_psd)

    def test_rebinSignal(self): 
        res, t, f = self.interf.runSimulation(1, verbosity=0)

        res_bin, f_bin = self.interf.rebinSignal(res, f, self.nbins_add, final_bin=True) 

        num = np.ceil(self.interf.instrumentDict["nf_ch"] / self.nbins_add)
        shape_bin = (t.size, num)
        self.assertEqual(res_bin["signal"].shape, shape_bin)
        
        res_bin, f_bin = self.interf.rebinSignal(res, f, self.nbins_add, final_bin=False) 

        num = np.floor(self.interf.instrumentDict["nf_ch"] / self.nbins_add)
        shape_bin = (t.size, num)
        self.assertEqual(res_bin["signal"].shape, shape_bin)
    
    def test_avgDirectSubtract(self): 
        res, t, f = self.interf.runSimulation(1, verbosity=0)
        res_red = self.interf.avgDirectSubtract(res)

        self.assertEqual(res_red.size, self.interf.instrumentDict["nf_ch"])

    @classmethod
    def tearDownClass(self):
        pass

if __name__ == "__main__":
    import nose2
    nose2.main()


