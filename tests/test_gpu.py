import unittest
from multiprocessing import Process
import gateau as gt

class TestGPU(unittest.TestCase):
    def test_selftest(self, for_real=False):
        """
        Run gateau's selftest and make sure it passes.
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
                )

            self.assertNotEqual(
                proc.exitcode,
                231,
                "GPUAssert error! Likely because of incorrect cuda setup, "
                "go fix it!!"
                )

            self.assertEqual(
                proc.exitcode,
                0,
                "Unexpected exit code!! No clue what caused this, "
                "good luck."
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

