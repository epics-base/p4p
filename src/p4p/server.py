
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
    """Server(conf=None, useenv=True, providers=[""])

    Run a PVAccess server serving Channels from the listed providers

    >>> S = Server(providers=["example"])
    >>> # do something else
    >>> S.stop()

    :param dict conf: Configuration keys for the server.  Uses same names as environment variables (aka. EPICS_PVAS_*)
    :param bool useenv: Whether to use process environment in addition to provided config.
    :param providers: A list of provider names or instances.

    When configuring a Server, conf keys provided to the constructor have the same name as the environment variables.
    If both are given, then the provided conf dict is used.

    Call Server.conf() to see a list of valid server (EPICS_PVAS_*) key names.

    The providers list must be a list of name strings (cf. installProvider()),
    or a list of Provider instances.  A mixture is not yet supported.
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
