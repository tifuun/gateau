from gateau import types
from gateau import bindings
from gateau import output_utils
from gateau import source_utils
from gateau import atmosphere_utils
from gateau.selftest import selftest

# __all__ controls what `from gateau import *` actually does
# but here we specify it to tell mypy
# that everything here is imported for namespace flattening,
# and not just on accident

__all__ = [
    'bindings',
    'output_utils',
    'source_utils',
    'atmosphere_utils',
    'selftest',
    'types',
    ]


