
from __future__ import absolute_import

import logging
import warnings
_log = logging.getLogger(__name__)

from weakref import WeakSet
from functools import partial

import cothread
from cothread import Spawn, Callback, Event, WaitForAll
from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
    'Handler',
)

# Spawn and Event have __slots__ which doesn't include '__weakref__'

class WeakSpawn(Spawn):
    __slots__=['__weakref__']

class WeakEvent(Event):
    __slots__=['__weakref__']

# a set of objects having a method 'Wait()'
_handlers = WeakSet()

def _sync():
    """I will wait until all pending handlers cothreads have completed
    """
    evt = WeakEvent(auto_reset=False)

    # first ensure that any pending callbacks from worker threads have been delivered
    # these are calls of _fromMain()
    Callback(evt.Signal)
    evt.Wait()

    evt.Reset() # reuse

    wait4 = set(_handlers)
    _handlers.clear()
    # use 'evt' as a proxy for all of the cothreads I'm waiting on,
    # so that later calls to _sync() will wait for these as well.
    _handlers.add(evt)

    WaitForAll(wait4)

    evt.Signal()
    _handlers.remove(evt)

# Callback() runs me in the main thread
def _fromMain(_handle, op, M, args):
    co = WeakSpawn(_handle, op, M, args)
    _handlers.add(co)

def _handle(op, M, args):
    try:
        M(*args)
        return
    except RemoteError as e:
        pass
    except Exception as e:
        _log.exception("Unexpected")
    finally:
        M = args = None # prevent lingering references after _sync() returns
    if op is not None:
        op.done(error=str(e))


class SharedPV(_SharedPV):

    def __init__(self, queue=None, **kws):
        _SharedPV.__init__(self, **kws)
        self._queue = queue or Callback

    def _exec(self, op, M, *args):
        self._queue(_fromMain, _handle, op, M, args)

    def close(self, destroy=False):
        _SharedPV.close(self, destroy)
        if destroy:
            _sync()
