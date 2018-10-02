
import logging
import warnings
_log = logging.getLogger(__name__)

import atexit
from functools import partial
from threading import Thread, Event

from ..util import ThreadedWorkQueue
from .raw import SharedPV as _SharedPV, Handler
from ..client.thread import RemoteError

__all__ = (
    'SharedPV',
    'Handler',
)

# lazy create a default work queues


class _DefaultWorkQueue(object):

    def __init__(self, workers=4):  # TODO: configurable?
        self.W = [None]*workers
        self.n = 0

    def __del__(self):
        self.stop()

    def __call__(self):
        W = self.W[self.n]
        if W is None:
            #  daemon=True  otherwise the MainThread exit handler tries to join too early
            W = self.W[self.n] = ThreadedWorkQueue(maxsize=0, daemon=True).start()

        # sort of load balancing by giving different queues to each SharedPV
        # but preserve ordering or callbacks as each SharedPV has only one queue
        self.n = (self.n+1)%len(self.W)
        return W

    def sync(self):
        [W.sync() for W in self.W if W is not None]

    def stop(self):
        [W.stop() for W in self.W if W is not None]
        self.W = [None]*len(self.W)

_defaultWorkQueue = _DefaultWorkQueue()

atexit.register(_defaultWorkQueue.stop)


def _on_queue(op, M, *args):
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

    """Shared state Process Variable.  Callback based implementation.

    .. note:: if initial=None, the PV is initially **closed** and
              must be :py:meth:`open()`'d before any access is possible.

    :param handler: A object which will receive callbacks when eg. a Put operation is requested.
                    May be omitted if the decorator syntax is used.
    :param Value initial: An initial Value for this PV.  If omitted, :py:meth:`open` s must be called before client access is possible.
    :param nt: An object with methods wrap() and unwrap().  eg :py:class:`p4p.nt.NTScalar`.
    :param callable wrap: As an alternative to providing 'nt=', A callable to transform Values passed to open() and post().
    :param callable unwrap: As an alternative to providing 'nt=', A callable to transform Values returned Operations in Put/RPC handlers.
    :param WorkQueue queue: The threaded :py:class:`WorkQueue` on which handlers will be run.
    :param dict options: A dictionary of configuration options.

    Creating a PV in the open state, with no handler for Put or RPC (attempts will error). ::

        from p4p.nt import NTScalar
        pv = SharedPV(nt=NTScalar('d'), value=0.0)
        # ... later
        pv.post(1.0)

    The full form of a handler object is: ::

        class MyHandler:
            def put(self, op):
                pass
            def rpc(self, op):
                pass
            def onFirstConnect(self): # may be omitted
                pass
            def onLastDisconnect(self): # may be omitted
                pass
        pv = SharedPV(MyHandler())

    Alternatively, decorators may be used. ::

        pv = SharedPV()
        @pv.put
        def onPut(pv, op):
            pass

    """

    def __init__(self, queue=None, **kws):
        _SharedPV.__init__(self, **kws)
        self._queue = queue or _defaultWorkQueue()
        self._disconnected = Event()
        self._disconnected.set()

    def _exec(self, op, M, *args):
        self._queue.push(partial(_on_queue, op, M, *args))

    def _onFirstConnect(self, _junk):
        self._disconnected.clear()

    def _onLastDisconnect(self, _junk):
        self._disconnected.set()

    def close(self, destroy=False, sync=False, timeout=None):
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
            # TODO: still not syncing PVA workers...
            self._queue.sync()
            self._disconnected.wait()
