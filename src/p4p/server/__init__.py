
import logging, warnings
_log = logging.getLogger(__name__)

import time
import atexit

from .._p4p import (Server as _Server,
                   installProvider,
                   removeProvider,
                   clearProviders,
                   StaticProvider as _StaticProvider,
                   DynamicProvider as _DynamicProvider,
                   ServerOperation,
                   )

__all__ = (
        'Server',
        'installProvider',
        'removeProvider',
        'StaticProvider',
        'DynamicProvider',
        'ServerOperation',
)

atexit.register(clearProviders)

class Server(object):
    """Server(conf=None, useenv=True, providers=[""])

    Run a PVAccess server serving Channels from the listed providers. ::

        S = Server(providers=["example"])
        # do something else
        S.stop()

    :param dict conf: Configuration keys for the server.  Uses same names as environment variables (aka. EPICS_PVAS_*)
    :param bool useenv: Whether to use process environment in addition to provided config.
    :param providers: A list of provider names or instances.

    When configuring a Server, conf keys provided to the constructor have the same name as the environment variables.
    If both are given, then the provided conf dict is used.

    Call Server.conf() to see a list of valid server (EPICS_PVAS_*) key names.

    The providers list must be a list of name strings (cf. installProvider()),
    or a list of Provider instances.  A mixture is not yet supported.

    As a convenience, a Server may be used as a context manager to automatically stop. ::

        with Server(providers=["example"]) as S:
        # do something else
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

    @classmethod
    def forever(klass, *args, **kws):
        """Create a server and block the calling thread until KeyboardInterrupt.
        Shorthand for: ::

            with Server(*args, **kws):
                try;
                    time.sleep(99999999)
                except KeyboardInterrupt:
                    pass
        """
        with klass(*args, **kws):
            _log.info("Running server")
            try:
                while True:
                    time.sleep(100)
            except KeyboardInterrupt:
                pass
            finally:
                _log.info("Stopping server")

class StaticProvider(_StaticProvider):
    """A channel provider which servers from a clearly defined list of names.
    This list may change at any time.
    """

class DynamicProvider(_DynamicProvider):
    """A channel provider which does not maintain a list of provided channel names.

       The following example shows a simple case, in fact so simple that StaticProvider
       is a better fit. ::
    
            class DynHandler(object):
                def __init__(self):
                    self.pv = SharedPV()
                def testChannel(self, name):
                    return name=="blah"
                def makeChannel(self, name, peer):
                    assert name=="blah"
                    return self.pv
            provider = DynamicProvider("arbitrary", DynHandler())
            server = Server(providers=[provider])
    """
    def __init__(self, name, handler):
        _DynamicProvider.__init__(self, name, self._WrapHandler(handler))

    class _WrapHandler(object):
        "Wrapper around user Handler which logs exception"
        def __init__(self, real):
            self._real = real
        def testChannel(self, name):
            try:
                return self._real.testChannel(name)
            except:
                _log.exception("Unexpected")
        def makeChannel(self, name, peer):
            try:
                return self._real.makeChannel(name, peer)
            except:
                _log.exception("Unexpected")
