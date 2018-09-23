
import logging
import warnings
_log = logging.getLogger(__name__)

from functools import partial

import asyncio

from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
        'Handler',
)


def _log_err(V):
    if isinstance(V, Exception):
        _log.error("Unhandled from SharedPV handler: %s", V)
        # TODO: figure out how to show stack trace...
        # until then, propagate in the hope that someone else will
    return V


def _handle(loop, op, M, args):
    try:
        _log.debug('SERVER HANDLE %s %s %s', op, M, args)
        maybeco = M(*args)
        if asyncio.iscoroutine(maybeco):
            _log.debug('SERVER SCHEDULE %s', maybeco)
            task = loop.create_task(maybeco)
            task.add_done_callback(_log_err)
        return
    except RemoteError as e:
        pass
    except Exception as e:
        _log.exception("Unexpected")
    if op is not None:
        op.done(error=str(e))


class SharedPV(_SharedPV):

    def __init__(self, handler=None, loop=None, **kws):
        self.loop = loop or asyncio.get_event_loop()
        _SharedPV.__init__(self, handler=handler, **kws)
        self._handler.loop = self.loop

    def _exec(self, op, M, *args):
        self.loop.call_soon_threadsafe(partial(_handle, self.loop, op, M, args))
        # 3.5 adds asyncio.run_coroutine_threadsafe()
