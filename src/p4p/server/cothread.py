
from __future__ import absolute_import

import logging
import warnings
_log = logging.getLogger(__name__)

from functools import partial

import cothread
from cothread import Spawn, Event, ThreadedEventQueue
from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
    'Handler',
)

def _worker(Q):
    while True:
        op = M = args = None # prevent holding references to completed work while waiting
        op, M, args = Q.Wait()
        try:
            M(*args)
            continue
        except RemoteError as e:
            pass
        except Exception as e:
            _log.exception("Unexpected")
        if op is not None:
            op.done(error=str(e))

class SharedPV(_SharedPV):

    def __init__(self, **kws):
        _SharedPV.__init__(self, **kws)
        self._Q = ThreadedEventQueue()
        self._W = Spawn(_worker, self._Q)
        self._disconnect_evt = Event(auto_reset=False)
        self._disconnect_evt.Signal()

    def _exec(self, op, M, *args):
        self._Q.Signal((op, M, args))

    def _onFirstConnect(self):
        self._disconnect_evt.Reset()

    def _onLastDisconnect(self):
        self._disconnect_evt.Signal()

    def close(self, destroy=False):
        _log.debug("CLOSE!!! %s", destroy)
        _SharedPV.close(self, destroy=destroy)
        if destroy:
            self._disconnect_evt.Wait()
