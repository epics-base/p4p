
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

Shutdown = object()

@asyncio.coroutine
def _worker(loop, Q):
    while True:
        op = M = args = None # prevent holding references to completed work while waiting
        op, M, args = yield from Q.get()

        if op is Shutdown:
            return

        try:
            maybeco = M(*args)
            if asyncio.iscoroutine(maybeco):
                task = loop.create_task(maybeco)
                task.add_done_callback(_log_err)
            continue # on success, wait for next work
        except RemoteError as e:
            pass
        except Exception as e:
            _log.exception("Unexpected")
        finally:
            Q.task_done()
        # signal error to remote
        if op is not None:
            op.done(error=str(e))

class SharedPV(_SharedPV):

    def __init__(self, handler=None, loop=None, **kws):
        self.loop = loop or asyncio.get_event_loop()
        _SharedPV.__init__(self, handler=handler, **kws)
        self._handler.loop = self.loop
        self._Q = asyncio.Queue(loop=self.loop)
        self._W = self.loop.create_task(_worker(self.loop, self._Q))
        self._disconnect_evt = asyncio.Event(loop=self.loop)
        self._disconnect_evt.set()

    def _exec(self, op, M, *args):
        self.loop.call_soon_threadsafe(partial(self._Q.put_nowait, (op, M, args)))
        # 3.5 adds asyncio.run_coroutine_threadsafe()

    def close(self, destroy=False):
        _SharedPV.close(self, destroy=destroy)

    @asyncio.coroutine
    def wait_closed(self):
        "Waits until onLastDisconnect() has been started"
        yield from self._disconnect_evt.wait()

    def _onFirstConnect(self):
        self._disconnect_evt.clear()

    def _onLastDisconnect(self):
        self._disconnect_evt.set()

    @asyncio.coroutine
    def shutdown(self):
        """Equivalent to close(destroy=True) followed by "yield from wait_closed()"
        and additional cleanup of internal Tasks.
        Unlike close(), shutdown() is irreversible
        """
        self.close(destroy=True)
        yield from self.wait_closed()
        yield from self._Q.put((Shutdown, None, None))
        yield from self._W
