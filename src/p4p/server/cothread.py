
from __future__ import absolute_import

import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial

import cothread
from cothread import Spawn, Callback
from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
    'Handler',
)

def _handle(op, M, *args):
    try:
        M(*args)
        return
    except RemoteError as e:
        pass
    except Exception as e:
        _log.exception("Unexpected")
    if op is not None:
        op.done(error=str(e))

class SharedPV(_SharedPV):
    def __init__(self, queue=None, **kws):
        _SharedPV.__init__(self, **kws)
        self._queue = queue or Callback

    def _exec(self, op, M, *args):
        self._queue(Spawn, _handle, op, M, *args)
