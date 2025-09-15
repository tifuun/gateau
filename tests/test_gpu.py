import unittest
from multiprocessing import Process
import gateau as gt

class TestGPU(unittest.TestCase):
    def test_selftest(self, for_real=False):
        """
        Run gateau's selftest and make sure it passes...

        Except Arend's "minimal test" actually crashes with a GPUAssert,
        so instead we test for that.
        """

        # TODO selftest tempdir
        # won't actually clean up if exit due to CUDA errror??

        if for_real is False:
            proc = Process(
                target=self.test_selftest,
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

        gt.selftest()

if __name__ == '__main__':
    unittest.main()

