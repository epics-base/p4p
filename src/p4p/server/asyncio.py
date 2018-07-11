
import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial

import asyncio

from .raw import SharedPV as _SharedPV

__all__ = (
        'SharedPV',
)

def _handle(loop, op, M, args):
    try:
        task = asyncio.ensure_future(M(*args))
    except Exception as e:
        if op is not None:
            op.done(error=str(e))
        _log.exception("Unexpected")

class SharedPV(_SharedPV):
    def __init__(self, handler=None, loop=None, **kws):
        self.loop = loop or asyncio.get_event_loop()
        _SharedPV.__init__(self, handler=handler, **kws)
        self._handler.loop = self.loop

    def _exec(self, op, M, *args):
        self.loop.call_soon_threadsafe(partial(_handle, self.loop, op, M, args))
        # 3.5 adds asyncio.run_coroutine_threadsafe()
