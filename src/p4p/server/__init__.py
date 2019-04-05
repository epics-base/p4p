import sys
import logging
import warnings
import re
import time
import uuid

from weakref import WeakSet

from .._p4p import (Server as _Server,
                    installProvider,
                    removeProvider,
                    clearProviders,
                    StaticProvider as _StaticProvider,
                    DynamicProvider as _DynamicProvider,
                    ServerOperation,
                    )

if sys.version_info >= (3, 0):
    unicode = str

_log = logging.getLogger(__name__)

__all__ = (
    'Server',
        'installProvider',
        'removeProvider',
        'StaticProvider',
        'DynamicProvider',
        'ServerOperation',
)

class Server(object):

    """Server(conf=None, useenv=True, providers=[""])

    :param providers: A list of provider names or instances.  See below.
    :param dict conf: Configuration keys for the server.  Uses same names as environment variables (aka. EPICS_PVAS_*)
    :param bool useenv: Whether to use process environment in addition to provided config.
    :param bool isolate: If True, override conf= and useenv= to select a configuration suitable for isolated testing.
                         eg. listening only on localhost with a randomly chosen port number.  Use `conf()` to determine
                         which port is being used.

    Run a PVAccess server serving Channels from the listed providers.
    The server is running after construction, until stop(). ::

        S = Server(providers=["example"])
        # do something else
        S.stop()

    As a convenience, a Server may be used as a context manager to automatically `stop()`. ::

        with Server(providers=["example"]) as S:
            # do something else

    When configuring a Server, conf keys provided to the constructor have the same name as the environment variables.
    If both are given, then the provided conf dict is used.

    Call Server.conf() to see a list of valid server (EPICS_PVAS_*) key names and the actual values.

    The providers list must be a list of name strings (cf. installProvider()),
    a list of `StaticProvider` or `DynamicProvider` instances, or dict "{'pv:name':`SharedPV`}" to implicitly creat a `StaticProvider`.
    """

    def __init__(self, providers, isolate=False, **kws):
        self.__keep_alive = [] # ick...

        if isinstance(providers, (bytes, unicode)):
            providers = providers.split() # split on space
            warnings.warn("Server providers list should be a list", DeprecationWarning)

        Ps = []
        for provider in providers:
            if isinstance(provider, (bytes, unicode)):
                if not re.match(r'^[^ \t\n\r]+$', provider):
                    raise ValueError("Invalid provider name: '%s'"%provider)
                Ps.append(provider)

            elif isinstance(provider, (_StaticProvider, _DynamicProvider)):
                Ps.append(provider)

            elif hasattr(provider, 'items'):
                P = StaticProvider()
                for name, pv in provider.items():
                    P.add(name, pv)
                Ps.append(P)
                # Normally user code is responsible for keeping the StaticProvider alive.
                # Not possible in this case though.
                self.__keep_alive.append(P)

            else:
                raise ValueError("providers=[] must be a list of string, SharedPV, or dict.  Not %s"%provider)

        if isolate:
            kws['useenv'] = False
            kws['conf'] = {
                'EPICS_PVAS_INTF_ADDR_LIST': '127.0.0.1',
                'EPICS_PVA_ADDR_LIST': '127.0.0.1',
                'EPICS_PVA_AUTO_ADDR_LIST': '0',
                'EPICS_PVA_SERVER_PORT': '0',
                'EPICS_PVA_BROADCAST_PORT': '0',
            }
        _log.debug("Starting Server isolated=%s, %s", isolate, kws)
        self._S = _Server(providers=Ps, **kws)

        _all_servers.add(self._S)

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.stop()

    def conf(self):
        """Return a dict() with the effective configuration this server is using.

        Suitable to pass to another Server to duplicate this configuration,
        or to a client Context to allow it to connect to this server. ::

            with Server(providers=["..."], isolate=True) as S:
                with p4p.client.thread.Context('pva', conf=S.conf(), useenv=False) as C:
                    print(C.get("pv:name"))
        """
        return self._S.conf()

    def stop(self):
        """Force server to stop serving, and close connections to existing clients.
        """
        _log.debug("Stopping Server")
        self._S.stop()
        self.__keep_alive = []

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

    :param str name: Provider name.  Must be unique within the local context in which it is used.
                     None, the default, will choose an appropriate value.
    """
    def __init__(self, name=None):
        if name is None:
            # Caller doesn't care.  Pick something unique w/o spaces
            name = str(uuid.uuid4())
        super(StaticProvider, self).__init__(name)

class DynamicProvider(_DynamicProvider):

    """A channel provider which does not maintain a list of provided channel names.

       The following example shows a simple case, in fact so simple that StaticProvider
       is a better fit. ::

            class DynHandler(object):
                def __init__(self):
                    self.pv = SharedPV()
                def testChannel(self, name): # return True, False, or DynamicProvider.NotYet
                    return name=="blah"
                def makeChannel(self, name, peer):
                    assert name=="blah"
                    return self.pv
            provider = DynamicProvider("arbitrary", DynHandler())
            server = Server(providers=[provider])
    """

    # Return from Handler.testChannel() to prevent caching of negative result.
    # Use when testChannel('name') might shortly return True
    NotYet = b'nocache'

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

_all_servers = WeakSet()

def _cleanup_servers():
    _log.debug("Stopping all Server instances")
    servers = list(_all_servers)
    for srv in servers:
        srv.stop()
