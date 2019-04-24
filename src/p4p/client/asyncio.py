
import logging
import warnings
import sys
_log = logging.getLogger(__name__)

import asyncio

from functools import partial, wraps

from . import raw
from .raw import Disconnected, RemoteError, Cancelled, Finished, LazyRepr
from ..wrapper import Value, Type
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = [
    'Context',
    'Value',
    'Type',
    'RemoteError',
    'timeout',
]


def timesout(deftimeout=5.0):
    """Decorate a coroutine to implement an overall timeout.

    The decorated coroutine will have an additional keyword
    argument 'timeout=' which gives a timeout in seconds,
    or None to disable timeout.

    :param float deftimeout: The default timeout= for the decorated coroutine.

    It is suggested perform one overall timeout at a high level
    rather than multiple timeouts on low-level operations. ::

        @timesout()
        @asyncio.coroutine
        def dostuff(ctxt):
            yield from ctxt.put('msg', 'Working')
            A, B = yield from ctxt.get(['foo', 'bar'])
            yield from ctxt.put('bar', A+B, wait=True)
            yield from ctxt.put('msg', 'Done')

        @asyncio.coroutine
        def exec():
            with Context('pva') as ctxt:
                yield from dostuff(ctxt, timeout=5)
    """
    def decorate(fn):
        assert asyncio.iscoroutinefunction(fn), "Place @timesout before @coroutine"

        @wraps(fn)
        @asyncio.coroutine
        def wrapper(*args, timeout=deftimeout, **kws):
            loop = kws.get('loop')
            fut = fn(*args, **kws)
            if timeout is None:
                yield from fut
            else:
                yield from asyncio.wait_for(fut, timeout=timeout, loop=loop)
        return wrapper
    return decorate


class Context(raw.Context):

    """
    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.
    :param conf dict: Configuration to pass to provider.  Depends on provider selected.
    :param bool useenv: Allow the provider to use configuration from the process environment.
    :param dict nt: Controls :ref:`unwrap`.  None uses defaults.  Set False to disable
    :param dict unwrap: Legacy :ref:`unwrap`.

    The methods of this Context will block the calling thread until completion or timeout

    The meaning, and allowed keys, of the configuration dictionary depend on the provider.

    The "pva" provider understands the following keys:

    * EPICS_PVA_ADDR_LIST
    * EPICS_PVA_AUTO_ADDR_LIST
    * EPICS_PVA_SERVER_PORT
    * EPICS_PVA_BROADCAST_PORT

    Timeout and Cancellation
    ^^^^^^^^^^^^^^^^^^^^^^^^

    All coroutines/Futures returned by Context methods can be cancelled.
    The methods of Context do not directly implement a timeout.
    Instead :py:meth:`asyncio.wait_for` should be used.
    It is suggested perform one overall timeout at a high level
    rather than multiple timeouts on low-level operations. ::

        @timesout()
        @asyncio.coroutine
        def dostuff(ctxt):
            yield from ctxt.put('msg', 'Working')
            A, B = yield from ctxt.get(['foo', 'bar'])
            yield from ctxt.put('bar', A+B, wait=True)
            yield from ctxt.put('msg', 'Done')

        @asyncio.coroutine
        def exec():
            with Context('pva') as ctxt:
                yield from dostuff(ctxt, timeout=5)
    """

    def __init__(self, provider, conf=None, useenv=True, nt=None, unwrap=None,
                 loop=None):
        super(Context, self).__init__(provider, conf=conf, useenv=useenv, nt=nt, unwrap=unwrap)
        self.loop = loop or asyncio.get_event_loop()

    @asyncio.coroutine
    def get(self, name, request=None):
        """Fetch current value of some number of PVs.

        :param name: A single name string or list of name strings
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A p4p.Value, or list of same.  Subject to :py:ref:`unwrap`.

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values. ::

            with Context('pva') as ctxt:
                V    = yield from ctxt.get('pv:name')
                A, B = yield from ctxt.get(['pv:1', 'pv:2'])
        """
        singlepv = isinstance(name, (bytes, str))
        if singlepv:
            return (yield from self._get_one(name, request=request))

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)

        futs = [self._get_one(N, request=R) for N, R in zip(name, request)]

        ret = yield from asyncio.gather(*futs, loop=self.loop)

        return ret

    @asyncio.coroutine
    def _get_one(self, name, request=None):
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).get(name, cb, request=request)

        _log.debug('get %s request=%s', name, request)
        try:
            return (yield from F)
        finally:
            op.close()

    @asyncio.coroutine
    def put(self, name, values, request=None, process=None, wait=None, get=True):
        """Write a new value of some number of PVs.

        :param name: A single name string or list of name strings
        :param values: A single value, a list of values, a dict, a `Value`.  May be modified by the constructor nt= argument.
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param str process: Control remote processing.  May be 'true', 'false', 'passive', or None.
        :param bool wait: Wait for all server processing to complete.
        :param bool get: Whether to do a Get before the Put.  If True then the value passed to the builder callable
                         will be initialized with recent PV values.  eg. use this with NTEnum to find the enumeration list.

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values

        If 'wait' or 'process' is specified, then 'request' must be omitted or None. ::

            with Context('pva') as ctxt:
                yield from ctxt.put('pv:name', 5.0)
                yield from ctxt.put(['pv:1', 'pv:2'], [1.0, 2.0])
                yield from ctxt.put('pv:name', {'value':5})

        The provided value(s) will be automatically coerced to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict, it is assumed to be a plain value
        and an attempt is made to store it in '.value' field.
        """
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]' % ('true' if wait else 'false', process or 'passive')

        singlepv = isinstance(name, (bytes, str))
        if singlepv:
            return (yield from self._put_one(name, values, request=request, get=get))

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)
        assert len(name) == len(values), (name, values)

        futs = [self._put_one(N, V, request=R, get=get) for N, V, R in zip(name, values, request)]

        yield from asyncio.gather(*futs, loop=self.loop)

    @asyncio.coroutine
    def _put_one(self, name, value, request=None, get=True):
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            _log.debug("put done %s %s", name, LazyRepr(value))
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).put(name, cb, builder=value, request=request, get=get)

        _log.debug('put %s <- %s request=%s', name, LazyRepr(value), request)
        try:
            value = yield from F
        finally:
            op.close()

    @asyncio.coroutine
    def rpc(self, name, value, request=None):
        """Perform a Remote Procedure Call (RPC) operation

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A Value.  Subject to :py:ref:`unwrap`.

        For example: ::

            uri = NTURI(['A','B'])
            with Context('pva') as ctxt:
                result = yield from ctxt.rpc('pv:name:add', uri.wrap('pv:name:add', 5, B=6))

        The provided value(s) will be automatically coerced to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict or Value, it is assumed to be a plain value
        and an attempt is made to store it in '.value' field.
        """
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).rpc(name, cb, value, request=request)

        try:
            return (yield from F)
        finally:
            op.close()

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
        assert asyncio.iscoroutinefunction(cb), "monitor callback must be coroutine"
        R = Subscription(name, cb, notify_disconnect=notify_disconnect, loop=self.loop)
        cb = partial(self.loop.call_soon_threadsafe, R._event)

        R._S = super(Context, self).monitor(name, cb, request)
        return R


class Subscription(object):

    """An active subscription.
    """

    def __init__(self, name, cb, notify_disconnect=False, loop=None):
        self.name, self._S, self._cb, self.loop = name, None, cb, loop
        self._notify_disconnect = notify_disconnect

        self._Q = asyncio.Queue(loop=self.loop)

        if notify_disconnect:
            self._Q.put_nowait(Disconnected())  # all subscriptions are inittially disconnected

        self._T = self.loop.create_task(self._handle())

    def __enter__(self):
        return self

    def __exit__(self, A, B, C):
        self.close()

    def close(self):
        """Begin closing subscription.
        """
        if self._S is not None:
            # after .close() self._event should never be called
            self._S.close()
            self._S = None
            self._Q.put_nowait(None)

    @property
    def done(self):
        'Has all data for this subscription been received?'
        return self._S is None or self._S.done()

    @property
    def empty(self):
        'Is data pending in event queue?'
        return self._S is None or self._S.empty()

    @asyncio.coroutine
    def wait_closed(self):
        """Wait until subscription is closed.
        """
        assert self._S is None, "Not close()'d"
        yield from self._T

    def _event(self, value):
        if self._S is not None:
            self._Q.put_nowait(value)

    @asyncio.coroutine
    def _handle(self):
        E = None
        try:
            while True:
                E = yield from self._Q.get()
                self._Q.task_done()
                _log.debug("Subscription %s handle %s", self.name, LazyRepr(E))

                S = self._S

                if isinstance(E, Cancelled):
                    return

                elif isinstance(E, Disconnected):
                    _log.debug('Subscription notify for %s with %s', self.name, E)
                    if self._notify_disconnect:
                        yield from self._cb(E)
                    else:
                        _log.info("Subscription disconnect %s", self.name)
                    continue

                elif isinstance(E, RemoteError):
                    _log.debug('Subscription notify for %s with %s', self.name, E)
                    if self._notify_disconnect:
                        yield from self._cb(E)
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
                    yield from self._cb(E)
                    i = (i + 1) % 4
                    if i == 0:
                        yield from asyncio.sleep(0)  # Not sure how necessary.  Ensure we go to the scheduler

                if S.done:
                    _log.debug('Subscription complete %s', self.name)
                    S.close()
                    self._S = None
                    if self._notify_disconnect:
                        E = Finished()
                        yield from self._cb(E)
                    break
        except:
            _log.exception("Error processing Subscription event: %s", LazyRepr(E))
            if self._S is not None:
                self._S.close()
            self._S = None
