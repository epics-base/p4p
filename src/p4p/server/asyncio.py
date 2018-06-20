
import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial

import asyncio

from .raw import SharedPV as _SharedPV

__all__ = (
        'SharedPV',
)

@asyncio.coroutine
def _handle(loop, op, M, args):
    try:
        yield from asyncio.ensure_future(M(*args)
    except Exception as e:
        if op is not None:
            op.done(error=str(e))
        _log.exception("Unexpected")

class SharedPV(_SharedPV):
    def __init__(self, handler=None, initial=None, loop=None):
        handler.loop = self.loop = loop or asyncio.get_event_loop()
        _SharedPV.__init__(self, handler=handler, initial=initial)
        self._queue = queue or Callback()

    def _exec(self, op, M, *args):
        fn = partial(_handle, loop, op, M, args)
        # 3.5 adds asyncio.run_coroutine_threadsafe()
        self.loop.call_soon_threadsafe(lambda:asyncio.ensure_future(fn, loop=self.loop))
