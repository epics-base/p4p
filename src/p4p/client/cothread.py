
from __future__ import absolute_import

import logging
import warnings
import sys
_log = logging.getLogger(__name__)

import cothread

from functools import partial

from . import raw
from .raw import Disconnected, RemoteError, Cancelled, Finished, LazyRepr
from .thread import TimeoutError
from ..wrapper import Value, Type
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = [
    'Context',
    'Value',
    'Type',
    'RemoteError',
]

if sys.version_info >= (3, 0):
    unicode = str


class Context(raw.Context):

    def get(self, name, request=None, timeout=5.0, throw=True):
        """Fetch current value of some number of PVs.

        :param name: A single name string or list of name strings
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.  If False then the Exception is returned instead of the Value

        :returns: A p4p.Value or Exception, or list of same.  Subject to :py:ref:`unwrap`.

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values

        >>> ctxt = Context('pva')
        >>> V = ctxt.get('pv:name')
        >>> A, B = ctxt.get(['pv:1', 'pv:2'])
        >>>
        """
        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            return self._get_one(name, request=request, timeout=timeout, throw=throw)

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)

        return cothread.WaitForAll(
            [cothread.Spawn(self._get_one, N, request=R, timeout=timeout, throw=throw,
                            raise_on_wait=True)
             for N, R in zip(name, request)]
        )

    def _get_one(self, name, request=None, timeout=5.0, throw=True):
        done = cothread.Event(auto_reset=False)

        def cb(value):
            assert not done, value # spurious second callback
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).get(name, cb, request=request)

        _log.debug('get %s request=%s', name, request)

        try:
            ret = done.Wait(timeout)
        except cothread.Timedout:
            ret = TimeoutError()
            if throw:
                raise ret
        finally:
            op.close()

        return ret

    def put(self, name, values, request=None, process=None, wait=None, timeout=5.0, get=True, throw=True):
        """Write a new value of some number of PVs.

        :param name: A single name string or list of name strings
        :param values: A single value, a list of values, a dict, a `Value`.  May be modified by the constructor nt= argument.
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value
        :param str process: Control remote processing.  May be 'true', 'false', 'passive', or None.
        :param bool wait: Wait for all server processing to complete.
        :param bool get: Whether to do a Get before the Put.  If True then the value passed to the builder callable
                         will be initialized with recent PV values.  eg. use this with NTEnum to find the enumeration list.

        :returns: A None or Exception, or list of same

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values

        If 'wait' or 'process' is specified, then 'request' must be omitted or None.

        >>> ctxt = Context('pva')
        >>> ctxt.put('pv:name', 5.0)
        >>> ctxt.put(['pv:1', 'pv:2'], [1.0, 2.0])
        >>> ctxt.put('pv:name', {'value':5})
        >>>

        The provided value(s) will be automatically coerced to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict, it is assumed to be a plain value
        and an attempt is made to store it in '.value' field.
        """
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]' % ('true' if wait else 'false', process or 'passive')

        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            return self._put_one(name, values, request=request, timeout=timeout, throw=throw, get=get)

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)
        assert len(name) == len(values), (name, values)

        return cothread.WaitForAll(
            [cothread.Spawn(self._put_one, N, V, request=R, timeout=timeout, throw=throw, get=get,
                            raise_on_wait=True)
             for N, V, R in zip(name, values, request)]
        )

    def _put_one(self, name, value, request=None, timeout=5.0, get=True, throw=True):
        done = cothread.Event(auto_reset=False)

        def cb(value):
            assert not done, value
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).put(name, cb, builder=value, request=request, get=get)

        _log.debug('put %s %s request=%s', name, LazyRepr(value), request)

        try:
            ret = done.Wait(timeout)
        except cothread.Timedout:
            ret = TimeoutError()
            if throw:
                raise ret
        finally:
            op.close()

        return ret

    def rpc(self, name, value, request=None, timeout=5.0, throw=True):
        """Perform a Remote Procedure Call (RPC) operation

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: A Value or Exception.  Subject to :py:ref:`unwrap`.

        >>> ctxt = Context('pva')
        >>> V = ctxt.rpc('pv:name:add', {'A':5, 'B'; 6})
        >>>

        The provided value(s) will be automatically coerced to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict, it is assumed to be a plain value
        and an attempt is made to store it in '.value' field.
        """
        done = cothread.Event(auto_reset=False)

        def cb(value):
            assert not done, value
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).rpc(name, cb, value, request=request)

        _log.debug('rpc %s %s request=%s', name, LazyRepr(value), request)

        try:
            try:
                ret = done.Wait(timeout)
            except cothread.Timedout:
                ret = TimeoutError()
            if throw and isinstance(ret, Exception):
                raise ret
        finally:
            op.close()

        return ret

    def monitor(self, name, cb, request=None, notify_disconnect=False):
        """Create a subscription.

        :param str name: PV name string
        :param callable cb: Processing callback
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param bool notify_disconnect: In additional to Values, the callback may also be call with instances of Exception.
                                       Specifically: Disconnected , RemoteError, or Cancelled
        :returns: a :py:class:`Subscription` instance

        The callable will be invoked with one argument which is either.

        * A p4p.Value (Subject to :py:ref:`unwrap`)
        * A sub-class of Exception (Disconnected , RemoteError, or Cancelled)
        """
        R = Subscription(name, cb, notify_disconnect=notify_disconnect)
        cb = partial(cothread.Callback, R._event)

        R._S = super(Context, self).monitor(name, cb, request)
        return R


class Subscription(object):

    def __init__(self, name, cb, notify_disconnect=False):
        self.name, self._S, self._cb = name, None, cb
        self._notify_disconnect = notify_disconnect

        self._Q = cothread.EventQueue()

        if notify_disconnect:
            self._Q.Signal(Disconnected())  # all subscriptions are inittially disconnected

        self._T = cothread.Spawn(self._handle)

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()

    def close(self):
        """Close subscription.
        """
        if self._S is not None:
            # after .close() self._event should never be called
            self._S.close()
            self._S = None
            self._Q.Signal(None)
            self._T.Wait()

    @property
    def done(self):
        'Has all data for this subscription been received?'
        return self._S is None or self._S.done()

    @property
    def empty(self):
        'Is data pending in event queue?'
        return self._S is None or self._S.empty()

    def _event(self, value):
        if self._S is not None:
            self._Q.Signal(value)

    def _handle(self):
        E = None
        try:
            while True:
                E = self._Q.Wait()
                _log.debug("Subscription %s handle %s", self.name, LazyRepr(E))

                S = self._S

                if isinstance(E, Cancelled):
                    return

                elif isinstance(E, Disconnected):
                    _log.debug('Subscription notify for %s with %s', self.name, E)
                    if self._notify_disconnect:
                        self._cb(E)
                    else:
                        _log.info("Subscription disconnect %s", self.name)
                    continue

                elif isinstance(E, RemoteError):
                    _log.debug('Subscription notify for %s with %s', self.name, E)
                    if self._notify_disconnect:
                        self._cb(E)
                    elif isinstance(E, RemoteError):
                        _log.error("Subscription Error %s", E)
                    return

                elif S is None:  # already close()'d
                    return

                i = 0
                while True:
                    E = S.pop()
                    if E is None or self._S is None:
                        break
                    self._cb(E)
                    i = (i + 1) % 4
                    if i == 0:
                        cothread.Yield()

                if S.done:
                    _log.debug('Subscription complete %s', self.name)
                    S.close()
                    self._S = None
                    if self._notify_disconnect:
                        E = Finished()
                        self._cb(E)
                    break
        except:
            _log.exception("Error processing Subscription event: %s", LazyRepr(E))
            self._S.close()
            self._S = None
