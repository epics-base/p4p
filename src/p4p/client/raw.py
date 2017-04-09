
import logging
_log = logging.getLogger(__name__)

import atexit, sys
from weakref import WeakSet

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from .._p4p import (Context as _Context,
                   Channel as _Channel)
from .._p4p import logLevelDebug

from ..wrapper import Value

__all__ = (
    'Context',
)

class Channel(_Channel):
    Value = Value
    name = property(_Channel.getName)

class Context(_Context):
    Channel = Channel

    def __init__(self, *args, **kws):
        _Context.__init__(self, *args, **kws)
        _all_contexts.add(self)
        # we keep strong refs here to shadow the Channel refs
        # from the underlying Provider, which we can't get at.
        self._channels = set()

    def close(self):
        if self._channels is not None:
            for ch in self._channels:
                ch.close()
            self._channels = None
        _Context.close(self)

    def channel(self, name):
        if self._channels is None:
            raise ValueError("Context closed")
        ch = _Context.channel(self, name)
        self._channels.add(ch)
        return ch

_all_contexts = WeakSet()

def _cleanup_contexts():
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()

atexit.register(_cleanup_contexts)
