from numpy import ndarray as arr
from typing import Sequence, Callable


ScanFuncFirst = Callable[[arr], arr]
ScanFuncStrict = Callable[[arr, float | arr, float | arr], arr]

ScanFuncStack = ScanFuncFirst | Sequence[ScanFuncStrict]

