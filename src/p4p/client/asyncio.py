
import logging
_log = logging.getLogger(__name__)

import asyncio

from functools import partial, wraps

from . import raw
from .raw import Disconnected, RemoteError, Cancelled, Finished
from ..wrapper import Value, Type
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = [
    'Context',
    'Value',
    'Type',
    'RemoteError',
    'timesout',
]

if hasattr(asyncio, 'get_running_loop'): # py >=3.7
    from asyncio import get_running_loop, create_task, all_tasks
else:
    from asyncio import _get_running_loop
    from asyncio.tasks import Task

    def get_running_loop():
        ret = _get_running_loop()
        if ret is None:
            raise RuntimeError('Thread has no running event loop')
        return ret

    def create_task(coro, *, name=None):
        return get_running_loop().create_task(coro)

    def all_tasks():
        return Task.all_tasks(loop=get_running_loop())

def timesout(deftimeout=5.0):
    """Decorate a coroutine to implement an overall timeout.

    The decorated coroutine will have an additional keyword
    argument 'timeout=' which gives a timeout in seconds,
    or None to disable timeout.

    :param float deftimeout: The default timeout= for the decorated coroutine.

    It is suggested to perform one overall timeout at a high level
    rather than multiple timeouts on low-level operations. ::

        @timesout()
        async def dostuff(ctxt):
            await ctxt.put('msg', 'Working')
            A, B = await ctxt.get(['foo', 'bar'])
            await ctxt.put('bar', A+B, wait=True)
            await ctxt.put('msg', 'Done')

        async def exec():
            with Context('pva') as ctxt:
                await dostuff(ctxt, timeout=5)
    """
    def decorate(fn):
        assert asyncio.iscoroutinefunction(fn), "Place @timesout before @coroutine"

        @wraps(fn)
        async def wrapper(*args, timeout=deftimeout, **kws):
            fut = fn(*args, **kws)
            if timeout is None:
                await fut
            else:
                await asyncio.wait_for(fut, timeout=timeout)
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
        async def dostuff(ctxt):
            await ctxt.put('msg', 'Working')
            A, B = await ctxt.get(['foo', 'bar'])
            await ctxt.put('bar', A+B, wait=True)
            await ctxt.put('msg', 'Done')

        async def exec():
            with Context('pva') as ctxt:
                await dostuff(ctxt, timeout=5)
    """

    def __init__(self, provider='pva', conf=None, useenv=True, nt=None, unwrap=None):
        super(Context, self).__init__(provider, conf=conf, useenv=useenv, nt=nt, unwrap=unwrap)

    async def get(self, name, request=None):
        """Fetch current value of some number of PVs.

        :param name: A single name string or list of name strings
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A p4p.Value, or list of same.  Subject to :py:ref:`unwrap`.

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values. ::

            with Context('pva') as ctxt:
                V    = await ctxt.get('pv:name')
                A, B = await ctxt.get(['pv:1', 'pv:2'])
        """
        singlepv = isinstance(name, (bytes, str))
        if singlepv:
            return (await self._get_one(name, request=request))

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)

        futs = [self._get_one(N, request=R) for N, R in zip(name, request)]

        ret = await asyncio.gather(*futs)

        return ret

    async def _get_one(self, name, request=None):
        F = asyncio.Future()

        def cb(value):
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, Exception):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(get_running_loop().call_soon_threadsafe, cb)

        op = super(Context, self).get(name, cb, request=request)

        _log.debug('get %s request=%s', name, request)
        try:
            return (await F)
        finally:
            op.close()

    async def put(self, name, values, request=None, process=None, wait=None, get=True):
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
                await ctxt.put('pv:name', 5.0)
                await ctxt.put(['pv:1', 'pv:2'], [1.0, 2.0])
                await ctxt.put('pv:name', {'value':5})

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
            return (await self._put_one(name, values, request=request, get=get))

        elif request is None:
            request = [None] * len(name)

        assert len(name) == len(request), (name, request)
        assert len(name) == len(values), (name, values)

        futs = [self._put_one(N, V, request=R, get=get) for N, V, R in zip(name, values, request)]

        await asyncio.gather(*futs)

    async def _put_one(self, name, value, request=None, get=True):
        F = asyncio.Future()

        def cb(value):
            _log.debug("put done %s %r", name, value)
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, Exception):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(get_running_loop().call_soon_threadsafe, cb)

        op = super(Context, self).put(name, cb, builder=value, request=request, get=get)

        _log.debug('put %s <- %r request=%s', name, value, request)
        try:
            value = await F
        finally:
            op.close()

    async def rpc(self, name, value, request=None):
        """Perform a Remote Procedure Call (RPC) operation

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.

        :returns: A Value.  Subject to :py:ref:`unwrap`.

        For example: ::

            uri = NTURI(['A','B'])
            with Context('pva') as ctxt:
                result = await ctxt.rpc('pv:name:add', uri.wrap('pv:name:add', 5, B=6))

        The provided value(s) will be automatically coerced to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict or Value, it is assumed to be a plain value
        and an attempt is made to store it in '.value' field.
        """
        F = asyncio.Future()

        def cb(value):
            if F.cancelled() or F.done():
                return  # ignore
            elif isinstance(value, Exception):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(get_running_loop().call_soon_threadsafe, cb)

        op = super(Context, self).rpc(name, cb, value, request=request)

        try:
            return (await F)
        finally:
            op.close()

    def monitor(self, name, cb, request=None, notify_disconnect=False) -> "Subscription":
        """Create a callback subscription.

        :param str name: PV name string
        :param callable cb: Processing callback
        :param request: A :py:class:`p4p.Value` or string to qualify this request, or None to use a default.
        :param bool notify_disconnect: In additional to Values, the callback may also be called with instances of Exception.
                                       Specifically: Disconnected , RemoteError, or Cancelled
        :returns: a :py:class:`Subscription` instance

        The callable will be invoked with one argument which is either.

        * A p4p.Value (Subject to :py:ref:`unwrap`)
        * A sub-class of Exception (Disconnected , RemoteError, or Cancelled)
        """
        assert asyncio.iscoroutinefunction(cb), "monitor callback must be coroutine"
        R = Subscription(name, cb, notify_disconnect=notify_disconnect)
        cb = partial(get_running_loop().call_soon_threadsafe, R._E.set)

        R._S = super(Context, self).monitor(name, cb, request)
        return R


class Subscription(object):

    """An active subscription.
    """

    def __init__(self, name, cb, notify_disconnect=False):
        self.name, self._S, self._cb = name, None, cb
        self._notify_disconnect = notify_disconnect

        self._run = True
        self._E = asyncio.Event()

        self._T = create_task(self._handle())

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
            self._run = False
            self._E.set()

    @property
    def done(self):
        'Has all data for this subscription been received?'
        return self._S is None or self._S.done()

    @property
    def empty(self):
        'Is data pending in event queue?'
        return self._S is None or self._S.empty()

    async def wait_closed(self):
        """Wait until subscription is closed.
        """
        assert self._S is None, "Not close()'d"
        await self._T

    async def _handle(self):
        if self._notify_disconnect:
            await self._cb(Disconnected())  # all subscriptions are inittially disconnected

        E = None
        try:
            while self._run:
                await self._E.wait()
                self._E.clear()
                _log.debug('Subscription %s wakeup', self.name)

                i = 0
                while self._run:
                    S = self._S
                    E = S.pop()
                    if E is None:
                        break

                    elif isinstance(E, Disconnected):
                        _log.debug('Subscription notify for %s with %s', self.name, E)
                        if self._notify_disconnect:
                            await self._cb(E)
                        else:
                            _log.info("Subscription disconnect %s", self.name)
                        continue

                    elif isinstance(E, RemoteError):
                        _log.debug('Subscription notify for %s with %s', self.name, E)
                        if self._notify_disconnect:
                            await self._cb(E)
                        elif isinstance(E, RemoteError):
                            _log.error("Subscription Error %s", E)
                        return

                    else:
                        await self._cb(E)

                    i = (i + 1) % 4
                    if i == 0:
                        await asyncio.sleep(0)  # Not sure how necessary.  Ensure we go to the scheduler

                    if S.done:
                        _log.debug('Subscription complete %s', self.name)
                        S.close()
                        self._S = None
                        if self._notify_disconnect:
                            E = Finished()
                            await self._cb(E)


        except asyncio.CancelledError:
            _log.debug("Cancelled Subscription: %r", self)
        except:
            _log.exception("Error processing Subscription event: %r", E)
        finally:
            if self._S is not None:
                self._S.close()
            self._S = None
