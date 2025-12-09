from numpy import ndarray as arr
from typing import Protocol, Sequence, Any, runtime_checkable, Tuple, Callable


ScanFuncFirst = Callable[[arr], arr]
ScanFuncStrict = Callable[[arr, float | arr, float | arr], arr]

ScanFuncStack = ScanFuncFirst | Sequence[ScanFuncStrict]

