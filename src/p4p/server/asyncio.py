
import logging
import warnings
_log = logging.getLogger(__name__)

from functools import partial
from weakref import WeakSet

import asyncio

from .raw import SharedPV as _SharedPV, Handler
from ..client.raw import LazyRepr
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
        'Handler',
)

@asyncio.coroutine
def _sync(loop):
    # wait until any pending callbacks are run
    evt = asyncio.Event(loop=loop)
    loop.call_soon(evt.set)
    yield from evt.wait()

    evt.clear() # reuse

    wait4 = set(loop._SharedPV_handlers) # snapshot of in-progress

    # like asyncio.wait() but non-invasive if further callbacks are added.
    # eg. like overlapping calls of _sync()
    # We're abusing the callback chain to avoid creating an Event for
    # each inprogress Future on the assumption that calls to _sync()
    # are relatively rare.

    cnt = len(wait4)
    fut = asyncio.Future(loop=loop)

    def _done(V):
        nonlocal cnt
        cnt -= 1
        if cnt==0 and not fut.cancelled():
            fut.set_result(None)
        return V # pass result along

    if cnt==0:
        fut.set_result(None)
    else:
        [W.add_done_callback(_done) for W in wait4]

    yield from fut

def _log_err(V):
    if isinstance(V, Exception):
        _log.error("Unhandled from SharedPV handler: %s", V)
        # TODO: figure out how to show stack trace...
        # until then, propagate in the hope that someone else will
    return V


def _handle(loop, op, M, args):
    try:
        _log.debug('SERVER HANDLE %s %s %s', op, M, LazyRepr(args))
        maybeco = M(*args)
        if asyncio.iscoroutine(maybeco):
            _log.debug('SERVER SCHEDULE %s', maybeco)
            task = loop.create_task(maybeco)
            task.add_done_callback(_log_err)
            task._log_destroy_pending = False # hack as we don't currently provide a way to join

            loop._SharedPV_handlers.add(task) # track in-progress
        return
    except RemoteError as e:
        err = e
    except Exception as e:
        _log.exception("Unexpected")
        err = e
    if op is not None:
        op.done(error=str(err))


class SharedPV(_SharedPV):

    def __init__(self, handler=None, loop=None, **kws):
        self.loop = loop or asyncio.get_event_loop()
        _SharedPV.__init__(self, handler=handler, **kws)
        self._handler.loop = self.loop
        self._disconnected = asyncio.Event(loop=self.loop)
        self._disconnected.set()
        if not hasattr(self.loop, '_SharedPV_handlers'):
            self.loop._SharedPV_handlers = WeakSet() # holds our in-progress Futures

    def _exec(self, op, M, *args):
        self.loop.call_soon_threadsafe(partial(_handle, self.loop, op, M, args))
        # 3.5 adds asyncio.run_coroutine_threadsafe()

    def _onFirstConnect(self, _junk):
        self._disconnected.clear()

    def _onLastDisconnect(self, _junk):
        self._disconnected.set()

    @asyncio.coroutine
    def _wait_closed(self):
        yield from _sync(self.loop)
        yield from self._disconnected.wait()

    def close(self, destroy=False, sync=False):
        """Close PV, disconnecting any clients.

        :param bool destroy: Indicate "permanent" closure.  Current clients will not see subsequent open().
        :param bool sync: When block until any pending onLastDisconnect() is delivered (timeout applies).
        :param float timeout: Applies only when sync=True.  None for no timeout, otherwise a non-negative floating point value.

        close() with destory=True or sync=True will not prevent clients from re-connecting.
        New clients may prevent sync=True from succeeding.
        Prevent reconnection by __first__ stopping the Server, removing with :py:method:`StaticProvider.remove()`,
        or preventing a :py:class:`DynamicProvider` from making new channels to this SharedPV.
        """
        _SharedPV.close(self, destroy)
        if sync:
            return self._wait_closed()
