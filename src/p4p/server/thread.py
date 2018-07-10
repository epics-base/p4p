
import logging, warnings
_log = logging.getLogger(__name__)

import atexit
from functools import partial
from threading import Thread

from ..util import WorkQueue
from .raw import SharedPV as _SharedPV

__all__ = (
    'SharedPV',
)

# lazy create a default work queue
class _DefaultWorkQueue(object):
    def __init__(self, workers=4): # TODO: configurable?
        self.Q = self.T = None
        self.N = workers
    def __del__(self):
        self.stop()
    def __call__(self):
        if self.Q is None:
            self.T = []
            self.Q = WorkQueue(maxsize=0)
            for _i in range(self.N):
                T = Thread(name="p4p.server.thread worker", target=self.Q.handle)
                T.daemon = True # otherwise the MainThread exit handler tries to join too early
                T.start()
                self.T.append(T)
        return self.Q
    def stop(self):
        if self.Q is None:
            return
        for T in self.T:
            self.Q.interrupt()
        for T in self.T:
            T.join()
        self.Q = self.T = None

_defaultWorkQueue = _DefaultWorkQueue()

atexit.register(_defaultWorkQueue.stop)

def _on_queue(op, M, *args):
    try:
        M(*args)
    except Exception as e:
        if op is not None:
            op.done(error=str(e))
        _log.exception("Unexpected")

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

    def _exec(self, op, M, *args):
        self._queue.push(partial(_on_queue, op, M, *args))
