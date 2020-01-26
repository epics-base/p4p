import logging

from PyQt5.QtCore import QObject, QCoreApplication, pyqtSignal, QEvent, QTimer

from . import raw
from .raw import Disconnected, RemoteError, Cancelled, Finished, LazyRepr
from ..wrapper import Value, Type
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
    success = pyqtSignal('PyQt_PyObject')
    error = pyqtSignal(str)

    def __init__(self, parent, timeout):
        QObject.__init__(self, parent)
        QTimer.singleShot(timeout*1000, self._timeout)

    @exceptionGuard
    def _timeout(self):
        if self._result is None:
            self._result = TimeoutError()
            self._notify()

    def _result(self, value):
        # called on PVA worker thread
        QCoreApplication.postEvent(self, CBEvent(value))

    @exceptionGuard
    def customEvent(self, evt):
        evt.accept()
        if self._result is None:
            self._result = evt.result
            self._notify()

    def _notify(self):
        if isinstance(evt.result, Exception):
            self.error.emit(str(evt.result))
        else:
            self.success.emit(evt.result)

class Subscription(QObject):
    _op = None
    update = pyqtSignal('PyQt_PyObject')
    error = pyqtSignal(str)

    def __init__(self, parent, cb, limitHz=10.0, notify_disconnect=False):
        QObject.__init__(self, parent)
        self._cb = cb or (lambda x:x)
        self._notify_disconnect = notify_disconnect
        self._active = None # timer id#
        self._holdoff = int(max(0.01, 1.0/limitHz)*1000)

    def _event(self, E):
        _log.debug('event1 %s', E)
        # called on PVA worker thread
        if isinstance(E, Cancelled):
            return

        QCoreApplication.postEvent(self, CBEvent(E))

    @exceptionGuard
    def customEvent(self, evt):
        _log.debug('event2 %s', evt)
        #evt.accept()
        E = evt.result

        if isinstance(E, RemoteError):
            self.error.emit(str(E))

        elif isinstance(E, Disconnected):
            if self._notify_disconnect:
                self.update.emit(self._cb(E))

            if self._active is not None:
                self.killTimer(self._active)
                self._active = None

        elif self._active is None:
            self._active = self.startTimer(self._holdoff)
            _log.debug('Start timer with %s', self._holdoff)

    @exceptionGuard
    def timerEvent(self, evt):
        V = self._op.pop()
        _log.debug('tick %s', V)

        if V is not None:
            try:
                V = self._cb(V)
            except Exception as E:
                _log.exception("Error in converter %s with %s", self._cb, V)
                self.error.emit(str(E))
            else:
                self.update.emit(V)

        elif self._active is not None:
            self.killTimer(self._active)
            self._active = None

class Context(raw.Context, QObject):
    """PyQt aware Context.
    """
    def __init__(self, provider, parent=None, **kws):
        raw.Context.__init__(self, provider, **kws)
        QObject.__init__(self, parent)

    # get() omitted (why would a gui want to do this?)

    def put(self, name, value, request=None, timeout=5.0,
            process=None, wait=None, get=True):
        """Begin put() operation
        
        Returns an Operation instance which will emit either a success and error signal.
        """
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]' % ('true' if wait else 'false', process or 'passive')

        raw_put = super(Context, self).put

        op = Operation(self, timeout)

        op._op = raw_put(name, op._result, builder=value, request=request, get=get)

        return op

    def rpc(self, name, value, request=None, timeout=5.0, throw=True):
        """Begin put() operation
        
        Returns an Operation instance which will emit either a success and error signal.
        """

        op = Operation(self)
        op._op = super(Context, self).rpc(name, op._result, value, request=request)
        return op

    def monitor(self, name, mangle=None, request=None, notify_disconnect=False, limitHz=10.0):
        """Returns a Subscription which will emit zero or more update signals, and perhaps an error signal.

        The mangle function, if provided will receive either a Value or an Exception, and returns
        the object actually emitted with the signal.

        limitHz, which must be provided, puts an upper limit on the rate at which the update signal will be emitted.
        """
        _log.debug('Subscribe to %s', name)
        op = Subscription(self, mangle, notify_disconnect=notify_disconnect, limitHz=limitHz)

        op._op = super(Context, self).monitor(name, op._event, request)

        return op
