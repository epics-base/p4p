
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

from .. import _p4p

from ..wrapper import Value, Type
from ..nt import _default_wrap, _default_unwrap

if sys.version_info >= (3, 0):
    unicode = str

__all__ = (
    'Subscription',
    'Context',
    'RemoteError',
)


class Cancelled(RuntimeError):

    "Cancelled from client (this) end."

    def __init__(self, msg=None):
        RuntimeError.__init__(self, msg or "Cancelled by client")


class Disconnected(RuntimeError):

    def __init__(self, msg=None):
        RuntimeError.__init__(self, msg or "Channel disconnected")


class Finished(Disconnected):

    def __init__(self, msg=None):
        Disconnected.__init__(self, msg or "Subscription Finished")


class RemoteError(RuntimeError):

    "Throw with an error message which will be passed back to the caller"


def unwrapHandler(handler, unwrap):
    """Wrap get/rpc handler to unwrap Value
    """
    def dounwrap(code, msg, val):
        _log.debug("Handler (%s, %s, %s) -> %s", code, msg, val, handler)
        try:
            if code == 0:
                handler(RemoteError(msg))
            elif code == 1:
                handler(Cancelled())
            else:
                if val is not None:
                    fn = unwrap.get(val.getID())
                    if fn:
                        val = fn(val)
                handler(val)
        except:
            _log.exception("Exception in Operation handler")
    return dounwrap


def monHandler(handler):
    def cb(code, msg):
        _log.debug("Update (%s, %s) -> %s", code, msg, handler)
        try:
            if code == 1:
                handler(RemoteError(msg))
            elif code == 2:
                handler(Cancelled())
            elif code == 4:
                handler(Disconnected())
            elif code == 8:
                handler(None)
        except:
            _log.exception("Exception in Monitor handler")
    return cb


def defaultBuilder(value):
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
            if isinstance(value, dict):
                for k, v in value.items():
                    V[k] = v
            else:
                V.value = value
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

    def __init__(self, context=None, unwrap=None, **kws):
        _log.debug("Subscription(%s)", kws)
        super(Subscription, self).__init__(**kws)
        self.context = context
        self._unwrap = unwrap or {}

    def pop(self):
        val = super(Subscription, self).pop()
        if val is not None:
            fn = self._unwrap.get(val.getID())
            if fn:
                val = fn(val)
        _log.debug("poll() -> %s", val)
        return val

    @property
    def done(self):
        return self.complete()

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()


class Context(object):

    """
    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.
    :param conf dict: Configuration to pass to provider.  Depends on provider selected.
    :param bool useenv: Allow the provider to use configuration from the process environment.
    :param dict unwrap: Controls :ref:`unwrap`.  None uses defaults.  Set False to disable
    """

    def __init__(self, provider=None, conf=None, useenv=None, unwrap=None, **kws):
        self.name = provider
        super(Context, self).__init__(**kws)

        if unwrap is None:
            self._unwrap = _default_unwrap
        elif not unwrap:
            self._unwrap = {}
        elif isinstance(unwrap, dict):
            self._unwrap = _default_unwrap.copy()
            self._unwrap.update(unwrap)
        else:
            raise ValueError("unwrap must be None, False, or dict, not %s" % unwrap)

        self._ctxt = None

        # initialize channel cache
        self.disconnect()

        self._ctxt = _p4p.ClientProvider(provider, conf=conf, useenv=useenv)

        _all_contexts.add(self)

    def close(self):
        if self._ctxt is None:
            return
        self._ctxt.close()
        self._ctxt = None
        self.disconnect()

        _all_contexts.discard(self)

    def __del__(self):
        if self._ctxt is not None:
            warnings.warn("%s collected without close()" % self.__class__)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()

    def _channel(self, name):
        # sub-class may wrap with some kind of lock to prevent the possibility of
        # extra channels.
        # extra channels should be avoided by ChannelProvider impls, but this isn't always the case.
        try:
            chan = self._channels[name]
        except KeyError:
            chan = _p4p.ClientChannel(self._ctxt, name)  # TODO: expose address and priority?
            self._channels[name] = chan
        return chan

    def disconnect(self, name=None):
        """Clear internal Channel cache, allowing currently unused channels to be implictly closed.

        :param str name: None, to clear the entire cache, or a name string to clear only a certain entry.
        """
        if name is None:
            self._channels = {}
        else:
            self._channels.pop(name)
        if self._ctxt is not None:
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
        chan = self._channel(name)
        return _p4p.ClientOperation(chan, handler=unwrapHandler(handler, self._unwrap),
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
        chan = self._channel(name)
        return _p4p.ClientOperation(chan, handler=unwrapHandler(handler, self._unwrap),
                                    builder=defaultBuilder(builder), pvRequest=wrapRequest(request), get=get, put=True)

    def rpc(self, name, handler, value, request=None):
        """Perform RPC operation on PV

        :param name: A single name string or list of name strings
        :param callable handler: Completion notification.  Called with a Value, RemoteError, or Cancelled
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A object with a method cancel() which may be used to abort the operation.
        """
        chan = self._channel(name)
        if value is None:
            value = Value(Type([]))
        return _p4p.ClientOperation(chan, handler=unwrapHandler(handler, self._unwrap),
                                    value=value, pvRequest=wrapRequest(request), rpc=True)

    def monitor(self, name, handler, request=None, **kws):
        """Begin subscription to named PV

        :param str name: PV name string
        :param callable handler: Completion notification.  Called with None (FIFO not empty), RemoteError, Cancelled, or Disconnected
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param bool notify_disconnect: Whether disconnect (and done) notifications are delivered to the callback (as None).

        :returns: A Subscription
        """
        chan = self._channel(name)
        return Subscription(context=self,
                            channel=chan, handler=monHandler(handler), pvRequest=wrapRequest(request),
                            unwrap=self._unwrap,
                            **kws)

# static methods
Context.providers = _p4p.ClientProvider.providers
Context.set_debug = _p4p.ClientProvider.set_debug
Context.makeRequest = _p4p.ClientProvider.makeRequest

_all_contexts = WeakSet()


def _cleanup_contexts():
    _log.debug("Closing all Client contexts")
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()
