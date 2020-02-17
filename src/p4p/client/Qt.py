import logging

from qtpy.QtCore import QObject, QCoreApplication, Signal, QEvent, QTimer

from . import raw
from .raw import Disconnected, RemoteError, Cancelled, Finished, LazyRepr
from ..wrapper import Value, Type
from .._p4p import serialize, ClientProvider
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = (
    'Context',
    'Value',
    'Type',
    'RemoteError',
    'TimeoutError',
)

_log = logging.getLogger(__name__)

# some pyqt callbacks are delicate, and will SIGSEGV is a python exception is allowed to propagate
def exceptionGuard(fn):
    def wrapper(*args, **kws):
        try:
            fn(*args, **kws)
        except:
            _log.exception('oops')
    return wrapper

class CBEvent(QEvent):
    # allocate an event ID#
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, result):
        QEvent.__init__(self, self.EVENT_TYPE)
        self.result = result

class Operation(QObject):
    _op = None
    _result = None
    # receives a Value or an Exception
    result = Signal(object)

    def __init__(self, parent, timeout):
        QObject.__init__(self, parent)
        self._active = self.startTimer(timeout*1000)

    def close(self):
        self._op.close()

    @exceptionGuard
    def timerEvent(self, evt):
        if self._result is None:
            self._result = TimeoutError()
            self.result.emit(self._result)

    def _result(self, value):
        # called on PVA worker thread
        QCoreApplication.postEvent(self, CBEvent(value))

    @exceptionGuard
    def customEvent(self, evt):
        if self._active is not None:
            self.killTimer(self._active)

        if self._result is None:
            self._result = evt.result
            self.result.emit(self._result)

class MCache(QObject):
    _op = None
    _last = None
    # receives a Value or an Exception
    update = Signal(object)

    def __init__(self, parent):
        QObject.__init__(self, parent)

        self._active = None
        self._holdoff = 10*1000 # acts as low limit on high limit

    def _add(self, slot, limitHz=10.0):
        holdoff = int(max(0.1, 1.0/limitHz)*1000)

        # Rate limiting for multiple consumers is hard.
        # We throttle to the highest consumer rate (shortest holdoff).
        if self._holdoff is None or self._holdoff > holdoff:
            self._holdoff = holdoff
            if self._active is not None:
                # restart timer
                self.killTimer(self._active)
                self._active = self.startTimer(self._holdoff)

        # TODO: re-adjust on slot disconnect?

        # schedule to receive initial update later (avoids recursion)
        QCoreApplication.postEvent(self, CBEvent(slot))

    def _event(self, E):
        _log.debug('event1 %s', E)
        # called on PVA worker thread
        if isinstance(E, Cancelled):
            return

        QCoreApplication.postEvent(self, CBEvent(E))

    @exceptionGuard
    def customEvent(self, evt):
        E = evt.result
        _log.debug('event2 %s', E)
        # E will be one of:
        #   None - FIFO not empty (call pop())
        #   RemoteError
        #   Disconnected
        #   some method, adding new subscriber

        if E is None:
            if self._active is None:
                self._active = self.startTimer(self._holdoff)
                _log.debug('Start timer with %s', self._holdoff)
            return

        elif isinstance(E, RemoteError):
            self._last = E
            self.update.emit(E)

        elif isinstance(E, Disconnected):
            self._last = E
            self.update.emit(E)

            if self._active is not None:
                self.killTimer(self._active)
                self._active = None

        else:
            E(self._last)
            self.update.connect(E)

    @exceptionGuard
    def timerEvent(self, evt):
        V = self._op.pop()
        _log.debug('tick %s', V)

        if V is not None:
            self._last = V
            self.update.emit(V)

        elif self._active is not None:
            self.killTimer(self._active)
            self._active = None

class Context(raw.Context):
    """PyQt aware Context.
    """
    def __init__(self, provider, parent=None, **kws):
        super(Context, self).__init__(provider, **kws)
        self._parent = QObject(parent)

        self._mcache = {}
        self._puts = {}

    # get() omitted (why would a gui want to do this?)

    def put(self, name, value, slot=None, request=None, timeout=5.0,
            process=None, wait=None, get=True):
        """Begin put() operation
        
        Returns an Operation instance which will emit either a success and error signal.
        """
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]' % ('true' if wait else 'false', process or 'passive')

        prev = self._puts.get(name)
        if prev is not None:
            # issuing new Put implicitly cancels any pending/queued Put
            prev.close()

        self._puts[name] = op = Operation(self._parent, timeout)
        if slot is not None:
            op.result.connect(slot)

        op._op = super(Context, self).put(name, op._result, builder=value, request=request, get=get)

        return op

    def rpc(self, name, value, slot, request=None, timeout=5.0, throw=True):
        """Begin put() operation
        
        Returns an Operation instance which will emit either a success and error signal.
        """

        op = Operation(self._parent, timeout)
        op.result.connect(slot)
        op._op = super(Context, self).rpc(name, op._result, value, request=request)
        return op

    def monitor(self, name, slot, request=None, limitHz=10.0):
        """Returns a Subscription which will emit zero or more update signals, and perhaps an error signal.

        The mangle function, if provided will receive either a Value or an Exception, and returns
        the object actually emitted with the signal.

        limitHz, which must be provided, puts an upper limit on the rate at which the update signal will be emitted.
        """
        _log.debug('Subscribe to %s with %s', name, request)
        if isinstance(request, (str, bytes)):
            request = ClientProvider.makeRequest(request)
        if isinstance(request, Value):
            request = serialize(request)

        key = (name, request) # (str, bytes|None)

        try:
            op = self._mcache[key]
        except KeyError:
            self._mcache[key] = op = MCache(self._parent)

            op._op = super(Context, self).monitor(name, op._event, request)

        op._add(slot, limitHz=limitHz)

        return op
