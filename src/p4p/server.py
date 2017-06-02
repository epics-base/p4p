
import logging, warnings
_log = logging.getLogger(__name__)

import atexit

from threading import Thread

from ._p4p import (Server as _Server,
                   installProvider,
                   removeProvider,
                   clearProviders,
                   RPCReply,
                   )

atexit.register(clearProviders)

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
        self.conf = self._S.conf
        self.stop = self._S.stop

    def __enter__(self):
        return self
    def __exit__(self, A, B, C):
        self.stop()

    def conf(self):
        """Return a dict() with the effective configuration this server is using.

        Suitable to pass to another Server to duplicate this configuration,
        or to a client Context to allow it to connect to this server.
        """
        pass

    def stop(self):
        """Force server to stop serving, and close connections to existing clients.
        """
        pass
