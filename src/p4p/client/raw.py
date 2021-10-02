
from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

import warnings
import sys
from weakref import WeakSet

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from .. import _p4p, set_debug as _set_debug
from .._p4p import Cancelled, Disconnected, Finished, RemoteError

from ..wrapper import Value, Type
from ..nt import buildNT

if sys.version_info >= (3, 0):
    unicode = str

__all__ = (
    'Subscription',
    'Context',
    'RemoteError',
)

class LazyRepr(object):
    """Log using repr()
    """
    __slots__ = ('value',)
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    __repr__ = __str__


def unwrapHandler(handler, nt):
    """Wrap get/rpc handler to unwrap Value
    """
    def dounwrap(code, msg, val, handler=handler):
        _log.debug("Handler (%s, %s, %s) -> %s", code, msg, LazyRepr(val), handler)
        try:
            if code == 0:
                handler(RemoteError(msg))
            elif code == 1:
                handler(Cancelled())
            else:
                if val is not None:
                    val = nt.unwrap(val)
                handler(val)
        except:
            _log.exception("Exception in Operation handler")
    return dounwrap


def monHandler(handler):
    def cb(handler=handler):
        _log.debug("Update %s", handler)
        try:
            handler()
        except:
            _log.exception("Exception in Monitor handler")
    return cb


def defaultBuilder(value, nt):
    """Reasonably sensible default handling of put builder
    """
    if callable(value):
        def logbuilder(V):
            try:
                value(V)
            except:
                _log.exception("Error in Builder")
                raise  # will be logged again
        return logbuilder

    def builder(V):
        try:
            if isinstance(value, Value):
                V[None] = value
            elif isinstance(value, dict):
                for k, v in value.items():
                    V[k] = v
            else:
                nt.assign(V, value)
        except:
            _log.exception("Exception in Put builder")
            raise  # will be printed to stdout from extension code.
    return builder


def wrapRequest(request):
    if request is None or isinstance(request, Value):
        return request
    return Context.makeRequest(request)


class Subscription(_p4p.ClientMonitor):

    """Interface to monitor subscription FIFO

    Use method poll() to try to pop an item from the FIFO.
    None indicates the FIFO is empty, must wait for another Data event before
    calling poll() again.

    complete()==True after poll()==False indicates that no more updates will
    ever be forthcoming.  This is normal (not error) completion.

    cancel() aborts the subscription.
    """

    def __init__(self, context, name, nt, **kws):
        _log.debug("Subscription(%s)", kws)
        super(Subscription, self).__init__(context, name, **kws)
        self.context = context
        self._nt = nt
        self.done = False

    def pop(self):
        val = super(Subscription, self).pop()
        assert val is None or isinstance(val, (Value, Exception)), val
        if isinstance(val, Value):
            val = self._nt.unwrap(val)
        elif isinstance(val, Finished):
            self.done = True
        _log.debug("poll() -> %s", LazyRepr(val))
        return val

    def complete(self):
        return self.done

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()

    if unicode is str:
        def __del__(self):
            self.close()

class Context(object):

    """
    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.
    :param conf dict: Configuration to pass to provider.  Depends on provider selected.
    :param bool useenv: Allow the provider to use configuration from the process environment.
    :param dict nt: Controls :ref:`unwrap`.  None uses defaults.  Set False to disable
    :param dict unwrap: Legacy :ref:`unwrap`.
    """

    def __init__(self, provider='pva', conf=None, useenv=None,
                 unwrap=None, nt=None,
                 **kws):
        self.name = provider
        super(Context, self).__init__(**kws)

        self._nt = buildNT(nt, unwrap)

        self._ctxt = None

        self._ctxt = _ClientProvider(provider, conf=conf, useenv=useenv)
        self.conf = self._ctxt.conf
        self.hurryUp = self._ctxt.hurryUp

        _all_contexts.add(self)

    makeRequest = _p4p.ClientProvider.makeRequest

    def close(self):
        if self._ctxt is None:
            return
        self._ctxt.close()
        self._ctxt = None

        _all_contexts.discard(self)

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()

    def disconnect(self, name=None):
        """Clear internal Channel cache, allowing currently unused channels to be implictly closed.

        :param str name: None, to clear the entire cache, or a name string to clear only a certain entry.
        """
        self._ctxt.disconnect(name)

    def _request(self, process=None, wait=None):
        """helper for building pvRequests

        :param str process: Control remote processing.  May be 'true', 'false', 'passive', or None.
        :param bool wait: Wait for all server processing to complete.
        """
        opts = []
        if process is not None:
            opts.append('process=%s' % process)
        if wait is not None:
            if wait:
                opts.append('wait=true')
            else:
                opts.append('wait=false')
        return 'field()record[%s]' % (','.join(opts))

    def get(self, name, handler, request=None):
        """Begin Fetch of current value of a PV

        :param name: A single name string or list of name strings
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param callable handler: Completion notification.  Called with a Value, RemoteError, or Cancelled

        :returns: A object with a method cancel() which may be used to abort the operation.
        """
        return _ClientOperation(self._ctxt, name, handler=unwrapHandler(handler, self._nt),
                                    pvRequest=wrapRequest(request), get=True, put=False)

    def put(self, name, handler, builder=None, request=None, get=True):
        """Write a new value to a PV.

        :param name: A single name string or list of name strings
        :param callable handler: Completion notification.  Called with None (success), RemoteError, or Cancelled
        :param callable builder: Called when the PV Put type is known.  A builder is responsible
                                 for filling in the Value to be sent.  builder(value)
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param bool get: Whether to do a Get before the Put.  If True then the value passed to the builder callable
                         will be initialized with recent PV values.  eg. use this with NTEnum to find the enumeration list.

        :returns: A object with a method cancel() which may be used to abort the operation.
        """
        return _ClientOperation(self._ctxt, name, handler=unwrapHandler(handler, self._nt),
                                    builder=defaultBuilder(builder, self._nt),
                                    pvRequest=wrapRequest(request), get=get, put=True)

    def rpc(self, name, handler, value, request=None):
        """Perform RPC operation on PV

        :param name: A single name string or list of name strings
        :param callable handler: Completion notification.  Called with a Value, RemoteError, or Cancelled
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A object with a method cancel() which may be used to abort the operation.
        """
        if value is None:
            value = Value(Type([]))
        return _ClientOperation(self._ctxt, name, handler=unwrapHandler(handler, self._nt),
                                    value=value, pvRequest=wrapRequest(request), rpc=True)

    def monitor(self, name, handler, request=None, **kws):
        """Begin subscription to named PV

        :param str name: PV name string
        :param callable handler: Completion notification.  Called with None (FIFO not empty), RemoteError, Cancelled, or Disconnected
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param bool notify_disconnect: Whether disconnect (and done) notifications are delivered to the callback (as None).

        :returns: A Subscription
        """
        return Subscription(self._ctxt, name,
                            nt=self._nt,
                            handler=monHandler(handler), pvRequest=wrapRequest(request),
                            **kws)

    @staticmethod
    def providers():
        return ["pva"]

    @staticmethod
    def set_debug(lvl):
        _set_debug(lvl)

set_debug = _p4p.logger_level_set

_all_contexts = WeakSet()


def _cleanup_contexts():
    _log.debug("Closing all Client contexts")
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()

class _ClientOperation(_p4p.ClientOperation):
    if unicode is str:
        def __del__(self):
            self.close()

class _ClientProvider(_p4p.ClientProvider):
    if unicode is str:
        def __del__(self):
            self.close()
