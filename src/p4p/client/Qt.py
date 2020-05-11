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
    """Context(provider, conf=None, useenv=True)
    
    PyQt aware Context.
    Methods in the class give notification of completion/update through Qt signals.

    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.
    :param dict conf: Configuration to pass to provider.  Depends on provider selected.
    :param bool useenv: Allow the provider to use configuration from the process environment.
    :param dict nt: Controls :ref:`unwrap`.  None uses defaults.  Set False to disable
    :param dict unwrap: Legacy :ref:`unwrap`.
    :param parent QObject: Parent for QObjects created through this Context.
    """
    def __init__(self, provider, parent=None, **kws):
        super(Context, self).__init__(provider, **kws)
        self._parent = QObject(parent)

        self._mcache = {}
        self._puts = {}


    def disconnect(self, name=None):
        if name is None:
            self._mcache = {}
            self._puts = {}
        else:
            self._mcache.pop(name)
            self._puts.pop(name)
        super(Context, self).disconnect(name)

    # get() omitted (why would a gui want to do this?)

    def put(self, name, value, slot=None, request=None, timeout=5.0,
            process=None, wait=None, get=True):
        """Write a new value to a single PV.

        Returns an Operation instance which will emit either a success and error signal.
        If the slot argument is provided, this will be connected in a race free way.

        The slot function will receive a python object which is either None (Success) or an Exception.

        Note that the returned Operation will also be stored internally by the Context.
        So the caller is not required to store it as well.
        This internal storage will only keep the most recent put() Operation for each PV name.
        A previous incomplete put() will be cancelled if/when put() is called again.

        :param name: A single name string or list of name strings
        :param values: A single value, a list of values, a dict, a `Value`.  May be modified by the constructor nt= argument.
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param slot: A callable object, such as a bound method, which can be passed to QObject.connect()
        :param float timeout: Operation timeout in seconds
        :param str process: Control remote processing.  May be 'true', 'false', 'passive', or None.
        :param bool wait: Wait for all server processing to complete.
        :param bool get: Whether to do a Get before the Put.  If True then the value passed to the builder callable
                         will be initialized with recent PV values.  eg. use this with NTEnum to find the enumeration list.
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
        """Perform a Remote Procedure Call (RPC) operation

        Returns an Operation instance which will emit either a success and error signal to the provided slot.
        This Operation instance must be stored by the caller or it will be implicitly cancelled.

        The slot function will receive a python object which is either a Value or an Exception.

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param slot: A callable object, such as a bound method, which can be passed to QObject.connect()
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: An Operation
        """

        op = Operation(self._parent, timeout)
        op.result.connect(slot)
        op._op = super(Context, self).rpc(name, op._result, value, request=request)
        return op

    def monitor(self, name, slot, request=None, limitHz=10.0):
        """Request subscription to named PV

        Request notification to the provided slot when a PV is updated.
        Subscriptions are managed by an internal cache,
        so than multiple calls to monitor() with the same PV name may be satisfied through a single subscription.

        limitHz, which must be provided, puts an upper limit on the rate at which the update signal will be emitted.
        Some update will be dropped in the PV updates more frequently.
        Reduction is done by discarding the second to last update.
        eg. It is guaranteed that the last update (present value) in the burst will be delivered.

        :param str name: PV name string
        :param callable cb: Processing callback
        :param slot: A callable object, such as a bound method, which can be passed to QObject.connect()
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param limitHz float: Maximum rate at which signals will be emitted.  In signals per second.
        :returns: a :py:class:`MCache` instance
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
