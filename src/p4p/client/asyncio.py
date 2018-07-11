
import logging, warnings, sys
_log = logging.getLogger(__name__)

import asyncio

from functools import partial

from . import raw
from .raw import Disconnected, RemoteError, Cancelled
from ..wrapper import Value, Type
from ..rpc import WorkQueue
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = [
    'Context',
    'Value',
    'Type',
    'RemoteError',
]


class Context(raw.Context):
    def __init__(self, provider, conf=None, useenv=True, unwrap=None,
                 loop=None):
        super(Context, self).__init__(provider, conf=conf, useenv=useenv, unwrap=unwrap)
        self.loop = loop or asyncio.get_event_loop()

    @asyncio.coroutine
    def get(self, name, request=None):
        singlepv = isinstance(name, (bytes, str))
        if singlepv:
            return (yield from self._get_one(name, request=request))

        elif request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)

        futs = [self._get_one(N, request=R) for N,R in zip(name, request)]

        ret = yield from asyncio.gather(futs, loop=self.loop)

        return ret

    @asyncio.coroutine
    def _get_one(self, name, request=None):
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            if F.cancelled() or F.done():
                return # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).get(name, cb,request=request)

        _log.debug('get %s request=%s', name, request)
        try:
            return (yield from F)
        finally:
            op.close()

    @asyncio.coroutine
    def put(self, name, values, request=None, process=None, wait=None):
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]'%('true' if wait else 'false', process or 'passive')

        singlepv = isinstance(name, (bytes, str))
        if singlepv:
            return (yield from self._put_one(name, values, request=request))

        elif request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)
        assert len(name)==len(values), (name, values)

        op = [self._put_one(N, V, request=R) for N, V, R in zip(name, values, request)]

        yield from asyncio.gather(futs, loop=self.loop)

    @asyncio.coroutine
    def _put_one(self, name, value, request=None):
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            _log.debug("put done %s %s", name, value)
            if F.cancelled() or F.done():
                return # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).put(name, cb, builder=value, request=request)

        _log.debug('put %s <- %s request=%s', name, value, request)
        try:
            value = yield from F
        finally:
            op.close()

    @asyncio.coroutine
    def rpc(self, name, value, request=None):
        F = asyncio.Future(loop=self.loop)

        def cb(value):
            if F.cancelled() or F.done():
                return # ignore
            elif isinstance(value, (RemoteError, Disconnected, Cancelled)):
                F.set_exception(value)
            else:
                F.set_result(value)
        cb = partial(self.loop.call_soon_threadsafe, cb)

        op = super(Context, self).rpc(name, cb, value, request=request)

        try:
            value = yield from F
        finally:
            op.close()

    def monitor(self, name, cb, request=None, notify_disconnect = False):
        assert asyncio.iscoroutinefunction(cb), "monitor callback must be coroutine"
        R = Subscription(name, cb, notify_disconnect=notify_disconnect, loop=self.loop)
        cb = partial(self.loop.call_soon_threadsafe, R._event)

        R._S = super(Context, self).monitor(name, cb, request)
        return R

class Subscription(object):
    def __init__(self, name, cb, notify_disconnect = False, loop = None):
        self.name, self._S, self._cb, self.loop = name, None, cb, loop
        self._notify_disconnect = notify_disconnect

        self._Q = asyncio.Queue(loop=self.loop)

        if notify_disconnect:
            self._Q.put_nowait(Disconnected()) # all subscriptions are inittially disconnected

        self._T = self.loop.create_task(self._handle())

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def close(self):
        if self._S is not None:
            # after .close() self._event should never be called
            self._S.close()
            self._S = None
            self._Q.put_nowait(None)

    @asyncio.coroutine
    def wait_closed(self):
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
                _log.debug("Subscription %s handle %s", self.name, E)

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

                elif S is None: # already close()'d
                    return

                i = 0
                while True:
                    E = S.pop()
                    if E is None or self._S is None:
                        break
                    yield from self._cb(E)
                    i = (i+1)%4
                    if i==0:
                        yield from asyncio.sleep(0) # Not sure how necessary.  Ensure we go to the scheduler

                if S.done:
                    _log.debug("Subscription complete")
                    S.close()
                    self._S = None
                    _log.debug('Subscription disconnect %s', self.name)
                    if self._notify_disconnect:
                        E = None
                        yield from self._cb(E)
                    break
        except:
            _log.exception("Error processing Subscription event: %s", E)
            self._S.close()
            self._S = None
