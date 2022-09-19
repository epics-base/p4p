
import logging

from functools import partial

import asyncio

from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError
from ..client.asyncio import get_running_loop, create_task, all_tasks

__all__ = (
    'SharedPV',
        'Handler',
)

_log = logging.getLogger(__name__)

def _log_err(V):
    if isinstance(V, Exception):
        _log.error("Unhandled from SharedPV handler: %s", V)
        # TODO: figure out how to show stack trace...
        # until then, propagate in the hope that someone else will
    return V


def _handle(pv, op, M, args): # callback in asyncio loop
    try:
        _log.debug('SERVER HANDLE %s %s %r', op, M, args)
        maybeco = M(*args)
        if asyncio.iscoroutine(maybeco):
            _log.debug('SERVER SCHEDULE %s', maybeco)
            task = create_task(maybeco)

            # we have no good place to join async put()/rpc() handler results
            # other than SharedPV.close(sync=True) which is both optional,
            # and potentially far in the future.  So we log and otherwise
            # discard the result.
            task.add_done_callback(_log_err)
            task._SharedPV = pv # mark so _wait_closed() can distinguish
        return # caller is responsible for op.done()
    except RemoteError as e:
        err = e
    except Exception as e:
        _log.exception("Unexpected")
        err = e
    if op is not None:
        op.done(error=str(err))


class SharedPV(_SharedPV):

    def __init__(self, handler=None, **kws):
        self.loop = get_running_loop()
        _SharedPV.__init__(self, handler=handler, **kws)
        self._disconnected = asyncio.Event()
        self._disconnected.set()

    def _exec(self, op, M, *args):
        # note than M may be _onFirstConnect or _onLastDisconnect
        self.loop.call_soon_threadsafe(partial(_handle, self, op, M, args))

    def _onFirstConnect(self, _junk):
        self._disconnected.clear()

    def _onLastDisconnect(self, _junk):
        self._disconnected.set()

    async def _wait_closed(self):
        """Wait until any in-progress asynchronous put()/rpc() handler tasks have completed.
        """
        _log.debug("Synchronizing %r", self)

        def _peak_done(F, V):
            F.set_result(V)
            return V

        Ts = []
        for t in all_tasks():
            if getattr(t, '_SharedPV', None) is not self:
                continue
            F = asyncio.Future()
            t.add_done_callback(partial(_peak_done, F))
            Ts.append(F)

        await asyncio.gather(*Ts, return_exceptions=True)
        # ignore any returned exceptions as they have already been logged

        _log.debug("Synchronized %r", self)

        # wait for Disconnect notification as well
        await self._disconnected.wait()

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
