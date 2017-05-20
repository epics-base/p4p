
import logging
_log = logging.getLogger(__name__)

from threading import Thread

from ._p4p import (Server as _Server,
                   installProvider,
                   removeProvider,
                   clearProviders,
                   RPCReply,
                   )

class Server(object):
    """Server(conf=None, useenv=True, providers="")

    Run a PVAccess server serving Channels from the listed providers

    >>> S = Server(providers="example")
    >>> S.start()
    >>> # do something else
    >>> S.stop()
    """
    def __init__(self, *args, **kws):
        self._S = _Server(*args, **kws)
        self._T = None
        self.conf = self._S.conf

    def __enter__(self):
        return self
    def __exit__(self, A, B, C):
        self.stop()

    def start(self):
        "Start running the PVA server"
        if self._T is not None:
            raise RuntimeError("Already running")
        self._T = Thread(target=self._S.run)
        _log.debug("Starting server thread")
        self._T.start()

    def stop(self):
        "Stop the server and block until this is done"
        T, self.T = self._T, None
        if T is not None:
            _log.debug("Stopping server thread")
            self._S.stop()
            _log.debug("Joining server thread")
            self._T.join()
            _log.debug("Joined server thread")
