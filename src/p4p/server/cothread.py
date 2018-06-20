
from __future__ import absolute_import

import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial

import cothread
from cothread import Spawn, Callback
from .raw import SharedPV as _SharedPV

__all__ = (
    'SharedPV',
)

def _handle(op, M, *args):
    try:
        M(*args)
    except Exception as e:
        if op is not None:
            op.done(error=str(e))
        _log.exception("Unexpected")

class SharedPV(_SharedPV):
    def __init__(self, handler=None, initial=None, queue=None):
        _SharedPV.__init__(self, handler=handler, initial=initial)
        self._queue = queue or Callback

    def _exec(self, op, M, *args):
        self._queue(Spawn, _handle, op, M, *args)
