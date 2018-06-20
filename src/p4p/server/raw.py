
import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial
import atexit

from threading import Thread

from .._p4p import (Server as _Server,
                   installProvider,
                   removeProvider,
                   clearProviders,
                   StaticProvider as _StaticProvider,
                   DynamicProvider as _DynamicProvider,
                   SharedPV as _SharedPV,
                   ServerOperation,
                   )

__all__ = (
        'Server',
        'installProvider',
        'removeProvider',
        'StaticProvider',
        'DynamicProvider',
        'SharedPV',
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

class SharedPV(_SharedPV):
    """Shared state Process Variable
    
    ... note: if initial=None, the PV is initially _closed_ and
        must be open()'d before any access is possible.

    :param handler: A object which will receive callbacks when eg. a Put operation is requested.
                    May be omitted if the decorator syntax is used.
    :param Value initial: An initial Value for this PV.  If omitted, open() must be called before client access is possible.

    The form of a handler object is: ::

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
    def __init__(self, handler=None, initial=None):
        self._handler = handler or self._DummyHandler()
        _SharedPV.__init__(self, self._WrapHandler(self._handler))
        if initial is not None:
            self.open(initial)

    class _DummyHandler(object):
        pass

    class _WrapHandler(object):
        "Wrapper around user Handler which logs exceptions"
        def __init__(self, real):
            self._real = real
        def onFirstConnect(self):
            try: # user handler may omit onFirstConnect()
                M = self._real.onFirstConnect
            except AttributeError:
                return
            try:
                M()
            except:
                _log.exception("Unexpected")
        def onLastDisconnect(self):
            try:
                M = self._real.onLastDisconnect
            except AttributeError:
                return
            try:
                M()
            except:
                _log.exception("Unexpected")
        def put(self, op):
            try:
                self._real.put(op)
            except Exception as e:
                op.done(error=str(e))
                _log.exception("Unexpected")
        def rpc(self, op):
            try:
                self._real.rpc(op)
            except Exception as e:
                op.done(error=str(e))
                _log.exception("Unexpected")

    @property
    def onFirstConnect(self):
        def decorate(fn):
            self._handler.onFirstConnect = partial(fn, self)
        return decorate
    @property
    def onLastDisconnect(self):
        def decorate(fn):
            self._handler.onLastDisconnect = partial(fn, self)
        return decorate
    @property
    def put(self):
        def decorate(fn):
            self._handler.put = partial(fn, self)
        return decorate
    @property
    def rpc(self):
        def decorate(fn):
            self._handler.rpc = partial(fn, self)
        return decorate

    def __repr__(self):
        return "SharedPV(open=%s)"%self.isOpen()
    __str__ = __repr__
