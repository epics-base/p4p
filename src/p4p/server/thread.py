
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
    def __init__(self, handler=None, initial=None, queue=None):
        _SharedPV.__init__(self, handler=handler, initial=initial)
        self._queue = queue or _defaultWorkQueue()

    def _exec(self, op, M, *args):
        self._queue.push(partial(_on_queue, op, M, *args))
