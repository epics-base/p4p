
import atexit
from weakref import WeakSet

from ._p4p import Context as _Context

__all__ = (
    'Context',
)

_all_contexts = WeakSet()

def _cleanup_contexts():
    contexts = list(_all_contexts)
    print("cleanup context", contexts)
    for ctxt in contexts:
        ctxt.close()

atexit.register(_cleanup_contexts)

class Context(_Context):
    def __init__(self, *args, **kws):
        _Context.__init__(self, *args, **kws)
        _all_contexts.add(self)
