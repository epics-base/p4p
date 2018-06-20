
import logging
_log = logging.getLogger(__name__)

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty
from threading import Thread

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
        self._Q.put_nowait(callable) # throws Queue.Full
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
                callable = self._Q.get(True, 1.0)
            except Empty:
                continue # retry on timeout
            try:
                if callable is self._stopit:
                    break
                callable()
            except:
                _log.exception("Error from WorkQueue")
            finally:
                self._Q.task_done()
