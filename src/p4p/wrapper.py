
import atexit

from ._p4p import Value as _Value
from ._p4p import Type
from ._p4p import clearProviders

atexit.register(clearProviders)

class Value(_Value):
    def __repr__(self):
        return '<Value: %s>'%self.tolist()
