
from __future__ import absolute_import

import logging, warnings, sys
_log = logging.getLogger(__name__)

import cothread

from functools import partial

from . import raw
from .raw import Disconnected, RemoteError, Cancelled
from .thread import TimeoutError
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

if sys.version_info>=(3,0):
    unicode = str

class Context(raw.Context):
    def get(self, name, request=None, timeout=5.0, throw=True):
        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            return self._get_one(name, request=request)

        elif request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)

        return cothread.WaitForAll(
            [cothread.Spawn(self._get_one, N, request=R, timeout=timeout, throw=throw,
                            raise_on_wait=True)
            for N,R in zip(name, request)]
        )

    def _get_one(self, name, request=None, timeout=5.0, throw=True):
        done = cothread.Event(auto_reset=False)

        def cb(value):
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).get(name, cb,request=request)

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

    def put(self, name, values, request=None, process=None, wait=None, timeout=5.0, throw=True):
        if request and (process or wait is not None):
            raise ValueError("request= is mutually exclusive to process= or wait=")
        elif process or wait is not None:
            request = 'field()record[block=%s,process=%s]'%('true' if wait else 'false', process or 'passive')

        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            return self._put_one(name, values, request=request, timeout=timeout, throw=throw)

        elif request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)
        assert len(name)==len(values), (name, values)

        return cothread.WaitForAll(
            [cothread.Spawn(self._put_one, N, V, request=R, timeout=timeout, throw=throw,
                            raise_on_wait=True)
            for N,V,R in zip(name, values, request)]
        )

    def _put_one(self, name, value, request=None, timeout=5.0, throw=True):
        done = cothread.Event(auto_reset=False)

        def cb(value):
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).put(name, cb, builder=value, request=request)

        _log.debug('put %s %s request=%s', name, value, request)

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
        done = cothread.Event(auto_reset=False)

        def cb(value):
            if isinstance(value, (RemoteError, Disconnected, Cancelled)):
                done.SignalException(value)
            else:
                done.Signal(value)

        cb = partial(cothread.Callback, cb)

        op = super(Context, self).rpc(name, cb, builder=value, request=request)

        _log.debug('rpc %s %s request=%s', name, value, value, request)

        try:
            ret = done.Wait(timeout)
        except cothread.Timedout:
            ret = TimeoutError()
            if throw:
                raise ret
        finally:
            op.close()

        return ret

    def monitor(self, name, cb, request=None, notify_disconnect = False):
        R = Subscription(name, cb, notify_disconnect=notify_disconnect)
        cb = partial(cothread.Callback, R._event)

        R._S = super(Context, self).monitor(name, cb, request)
        return R

class Subscription(object):
    def __init__(self, name, cb, notify_disconnect = False):
        self.name, self._S, self._cb = name, None, cb
        self._notify_disconnect = notify_disconnect

        self._Q = cothread.EventQueue()

        if notify_disconnect:
            self._Q.Signal(Disconnected()) # all subscriptions are inittially disconnected

        self._T = cothread.Spawn(self._handle)

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def close(self):
        if self._S is not None:
            # after .close() self._event should never be called
            self._S.close()
            self._S = None
            self._Q.Signal(None)
            self._T.Wait()

    def _event(self, value):
        if self._S is not None:
            self._Q.Signal(value)


    def _handle(self):
        E = None
        try:
            while True:
                E = self._Q.Wait()
                _log.debug("Subscription %s handle %s", self.name, E)

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

                elif S is None: # already close()'d
                    return

                i = 0
                while True:
                    E = S.pop()
                    if E is None or self._S is None:
                        break
                    self._cb(E)
                    i = (i+1)%4
                    if i==0:
                        cothread.Yield()

                if S.done:
                    _log.debug("Subscription complete")
                    S.close()
                    self._S = None
                    _log.debug('Subscription disconnect %s', self.name)
                    if self._notify_disconnect:
                        E = None
                        self._cb(E)
                    break
        except:
            _log.exception("Error processing Subscription event: %s", E)
            self._S.close()
            self._S = None
