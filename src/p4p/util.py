
import logging
_log = logging.getLogger(__name__)

from functools import partial

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty
from threading import Thread, Event

__all__ = [
    'WorkQueue',
]


class WorkQueue(object):

    """A threaded work queue.
    """
    _stopit = object()

    def __init__(self, maxsize=5):
        self._Q = Queue(maxsize=maxsize)

    def push(self, callable):
        self._Q.put_nowait(callable)  # throws Queue.Full

    def push_wait(self, callable):
        self._Q.put(callable)

    def interrupt(self):
        """Break one call to handle()

        eg. Call N times to break N threads.

        This call blocks if the queue is full.
        """
        self._Q.put(self._stopit)

    def handle(self):
        """Process queued work until interrupt() is called
        """
        while True:
            # TODO: Queue.get() (and anything using thread.allocate_lock
            #       ignores signals :(  so timeout periodically to allow delivery
            try:
                callable = None # ensure no lingering references to past work while blocking
                callable = self._Q.get(True, 1.0)
            except Empty:
                continue  # retry on timeout
            try:
                if callable is self._stopit:
                    break
                callable()
            except:
                _log.exception("Error from WorkQueue")
            finally:
                self._Q.task_done()

class ThreadedWorkQueue(WorkQueue):
    def __init__(self, name=None, workers=1, daemon=False, **kws):
        assert workers>=1, workers
        WorkQueue.__init__(self, **kws)
        self.name = name
        self._daemon = daemon
        self._T = [None]*workers

    def __enter__(self):
        self.start()
    def __exit__(self, A,B,C):
        self.stop()

    def start(self):
        for n in range(len(self._T)):
            if self._T[n] is not None:
                continue
            T = self._T[n] = Thread(name='%s[%d]'%(self.name, n), target=self.handle)
            T.daemon = self._daemon
            T.start()

        return self # allow chaining

    def stop(self):
        [self.interrupt() for T in self._T if T is not None]
        [T.join()      for T in self._T if T is not None]
        self._T = [None]*len(self._T)

        return self # allow chaining

    def sync(self, timeout=None):
        wait1 = [Event() for _n in range(len(self._T))]
        wait2 = [Event() for _n in range(len(self._T))]

        def syncX(wait1, wait2):
            wait1.set()
            wait2.wait()

        [self.push_wait(partial(syncX, wait1[n], wait2[n])) for n in range(len(self._T))]

        # wait for all workers to ready wait1 barrier
        for W in wait1:
            W.wait(timeout=timeout)

        # allow workers to proceeed
        for W in wait2:
            W.set()

        return self # allow chaining

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
