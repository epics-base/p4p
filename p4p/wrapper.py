
from ._p4p import Value as _Value
from ._p4p import Type

class Value(_Value):
    def __repr__(self):
        return '<Value: %s>'%self.tolist()
