
import atexit

from ._p4p import Value as _Value
from ._p4p import Type
from ._p4p import clearProviders

atexit.register(clearProviders)

class Value(_Value):
    """Value(type, value=None)

    Structured value container. Supports dict-list and object-list access

    :param Type type: A :py:class:`Type` describing the structure
    :param dict value: Initial values to populate the Value
    """
    def __repr__(self):
        return '<Value: %s>'%self.tolist()
    id = property(_Value.getID)
