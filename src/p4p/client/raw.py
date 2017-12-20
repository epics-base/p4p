
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
        self._channels = {}

    def close(self):
        self._channels.clear()
        _Context.close(self)
        _all_contexts.discard(self)

    def disconnect(self, name):
        self._channels.pop(name, None)

    def channel(self, name):
        try:
            return self._channels[name]
        except KeyError:
            self._channels[name] = ch = _Context.channel(self, name)
            return ch

_all_contexts = WeakSet()

def _cleanup_contexts():
    _log.debug("Closing all Client contexts")
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()

atexit.register(_cleanup_contexts)
