
import logging
_log = logging.getLogger(__name__)

from threading import Thread

from ._p4p import (Server as _Server,
                   installProvider,
                   removeProvider,
                   clearProviders,
                   )

class Server(object):
    def __init__(self, *args, **kws):
        self._S = _Server(*args, **kws)
        self._T = None

    def start(self):
        if self._T is not None:
            raise RuntimeError("Already running")
        self._T = Thread(target=self._S.run)
        _log.debug("Starting server thread")
        self._T.start()

    def stop(self):
        T, self.T = self._T, None
        if T is not None:
            _log.debug("Stopping server thread")
            self._S.stop()
            _log.debug("Joining server thread")
            self._T.join()
            _log.debug("Joined server thread")
