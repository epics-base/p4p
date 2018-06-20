
import logging, warnings
_log = logging.getLogger(__name__)

from functools import partial

from threading import Thread

from .._p4p import SharedPV as _SharedPV

__all__ = (
        'SharedPV',
)

class SharedPV(_SharedPV):
    """Shared state Process Variable.  Callback based implementation.
    
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
        self._whandler = self._WrapHandler(self, self._handler)
        _SharedPV.__init__(self, self._whandler)
        if initial is not None:
            self.open(initial)

    def _exec(self, op, M, *args): # sub-classes will replace this
        try:
            M(*args)
        except Exception as e:
            if op is not None:
                op.done(error=str(e))
            _log.exception("Unexpected")

    class _DummyHandler(object):
        pass

    class _WrapHandler(object):
        "Wrapper around user Handler which logs exceptions"
        def __init__(self, pv, real):
            self._pv = pv # this creates a reference cycle, which should be collectable since SharedPV supports GC
            self._real = real

        def onFirstConnect(self):
            try: # user handler may omit onFirstConnect()
                M = self._real.onFirstConnect
            except AttributeError:
                return
            self._pv._exec(None, M, self._pv)

        def onLastDisconnect(self):
            try:
                M = self._real.onLastDisconnect
            except AttributeError:
                return
            self._pv._exec(None, M, self._pv)

        def put(self, op):
            _log.debug('PUT %s %s', self._pv, op)
            try:
                self._pv._exec(op, self._real.put, self._pv, op)
            except AttributeError:
                op.done(error="Put not supported")

        def rpc(self, op):
            _log.debug('RPC %s %s', self._pv, op)
            try:
                self._pv._exec(op, self._real.rpc, self._pv, op)
            except AttributeError:
                op.done(error="RPC not supported")

    @property
    def onFirstConnect(self):
        def decorate(fn):
            self._handler.onFirstConnect = fn
        return decorate
    @property
    def onLastDisconnect(self):
        def decorate(fn):
            self._handler.onLastDisconnect = fn
        return decorate
    @property
    def put(self):
        def decorate(fn):
            self._handler.put = fn
        return decorate
    @property
    def rpc(self):
        def decorate(fn):
            self._handler.rpc = fn
        return decorate

    def __repr__(self):
        return "%s(open=%s)"%(self.__class__.__name__, self.isOpen())
    __str__ = __repr__
