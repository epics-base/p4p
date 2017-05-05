
from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

from itertools import izip
from functools import partial
import json, threading

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from . import raw
from ..wrapper import Value, Type
from ..rpc import WorkQueue
from .._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                    logLevelInfo, logLevelWarn, logLevelError,
                    logLevelFatal, logLevelOff)

__all__ = [
    'Context',
    'Value',
    'Type',
]

class Subscription(object):
    """An active subscription.
    """
    def __init__(self, ctxt, name, cb):
        self.name, self._S, self._cb = name, None, cb
        self._Q = ctxt._queue()
    def close(self):
        """Close subscription.
        """
        if self._S is not None:
            E = threading.Event()
            # after .close() self._event should never be called
            self._S.close()
            # now wait for any pending calls to self._handle
            # TODO: detect when called from worker and avoid deadlock
            self._Q.push_wait(E.set)
            E.wait()
            self._S = None
    @property
    def done(self):
        'Has all data for this subscription been received?'
        return self._S is None or self._S.done()
    @property
    def empty(self):
        'Is data pending in event queue?'
        return self._S is None or self._S.empty()
    def _event(self, E):
        try:
            assert self._S is not None, self._S
            #TODO: ensure ordering of error and data events
            _log.debug('Subscription wakeup for %s with %s', self.name, E)
            self._inprog = True
            self._Q.push(partial(self._handle, E))
        except:
            _log.exception("Lost Subscription update: %s", E)
    def _handle(self, E):
        try:
            if E is not None:
                self._cb(E)
                return
            while True:
                E = self._S.pop()
                if E is None:
                    break
                self._cb(E)
            if self._S.done():
                _log.debug("Subscription complete")
                self._S.close()
                self._S = None
                self._cb(None)
        except:
            _log.exception("Error processing Subscription event: %s", E)
            self._S.close()
            self._S = None

class Context(object):
    """Context(providerName, conf=None, useenv=True)

    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.

    The method of this Context will block the calling thread until completion or timeout
    """
    Value = Value

    providers = raw.Context.providers
    set_debug = raw.Context.set_debug

    def __init__(self, *args, **kws):
        self._Qmax = kws.pop('maxsize', 0)
        self._ctxt = raw.Context(*args, **kws)

        self._channels = {}

        # lazy start threaded WorkQueue
        self._Q, self._T = None, None

    def _queue(self):
        if self._Q is None:
            Q = WorkQueue(maxsize=self._Qmax)
            T = threading.Thread(name='p4p Context worker', target=Q.handle)
            T.start()
            self._Q, self._T = Q, T
        return self._Q

    def close(self):
        """Force close all Channels and cancel all Operations
        """
        if self._Q is not None:
            self._Q.interrupt()
            self._T.join()
            self._Q, self._T = None, None
        self._channels = None
        self._ctxt.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def _channel(self, name):
        try:
            return self._channels[name]
        except KeyError:
            self._channels[name] = ch = self._ctxt.channel(name)
            return ch

    def get(self, names, requests=None, timeout=5.0, throw=True):
        """Fetch current value of some number of PVs.
        
        :param names: A single name string or list of name strings
        :param requests: None or a Value to qualify this request
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.  If False then the Exception is returned instead of the Value

        :returns: A Value or Exception, or list of same

        When invoked with a single name then returns is a single value.
        When invoked with a list of names, then returns a list of values

        >>> ctxt = Context('pva')
        >>> V = ctxt.get('pv:name')
        >>> A, B = ctxt.get(['pv:1', 'pv:2'])
        >>>
        """
        singlepv = isinstance(names, (bytes, unicode))
        if singlepv:
            names = [names]
            if requests is not None:
                requests = [requests]

        if requests is None:
            requests = [None]*len(names)

        assert len(names)==len(requests), (names, requests)

        # use Queue instead of Event to allow KeyboardInterrupt
        done = Queue(maxsize=len(names))
        result = [None]*len(names)
        ops = [None]*len(names)

        try:
            for i,(name, req) in enumerate(izip(names, requests)):
                _log.debug('gext %s', name)
                ch = self._channel(name)
                def cb(value, i=i):
                    try:
                        done.put_nowait((value, i))
                    except:
                        _log.exception("Error queuing get result %s", value)
                _log.debug('get %s w/ %s', name, req)
                ops[i] = ch.get(cb, request=req)

            for _n in range(len(names)):
                value, i = done.get(timeout=timeout)
                _log.debug('got %s %s', names[i], value)
                if throw and isinstance(value, Exception):
                    raise value
                result[i] = value


            if singlepv:
                return result[0]
            else:
                return result
        except:
            [op and op.cancel() for op in ops]
            raise

    def put(self, names, values, requests=None, timeout=5.0, throw=True):
        """Write a new value of some number of PVs.
        
        :param names: A single name string or list of name strings
        :param values: A single value or a list of values
        :param requests: None or a Value to qualify this request
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: A None or Exception, or list of same

        When invoked with a single name then returns is a single value.
        When invoked with a list of names, then returns a list of values

        >>> ctxt = Context('pva')
        >>> ctxt.put('pv:name', 5.0)
        >>> ctxt.put(['pv:1', 'pv:2'], [1.0, 2.0])
        >>> ctxt.put('pv:name', {'value':5})
        >>>

        The provided value(s) will be automatically coearsed to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict, it is assumed to be a plan value
        and an attempt is made to store it in '.value' field.
        """
        singlepv = isinstance(names, (bytes, unicode))
        if singlepv:
            names = [names]
            values = [values]
            if requests is not None:
                requests = [requests]

        if requests is None:
            requests = [None]*len(names)

        assert len(names)==len(requests), (names, requests)
        assert len(names)==len(values), (names, values)

        # use Queue instead of Event to allow KeyboardInterrupt
        done = Queue(maxsize=len(names))
        result = [None]*len(names)
        ops = [None]*len(names)

        try:
            for i,(name, value, req) in enumerate(izip(names, values, requests)):
                if isinstance(value, (bytes, unicode)) and value[:1]=='{':
                    try:
                        value = json.loads(value)
                    except ValueError:
                        raise ValueError("Unable to interpret '%s' as json"%value)

                ch = self._channel(name)

                # callback to build PVD Value from PY value
                def vb(type, value=value, i=i):
                    try:
                        if isinstance(value, dict):
                            V = self.Value(type, value)
                        else:
                            V = self.Value(type, {})
                            V.value = value # will try to cast str -> *
                        return V
                    except Exception as E:
                        _log.exception("Error building put value %s", value)
                        done.put_nowait((E, i))
                        raise E

                # completion callback
                def cb(value, i=i):
                    try:
                        done.put_nowait((value, i))
                    except:
                        _log.exception("Error queuing put result %s", value)
                ops[i] = ch.put(cb, vb, request=req)

            for _n in range(len(names)):
                value, i = done.get(timeout=timeout)
                if throw and isinstance(value, Exception):
                    raise value
                result[i] = value

            if singlepv:
                return result[0]
            else:
                return result
        except:
            [op and op.cancel() for op in ops]
            raise

    def rpc(self, name, value, request=None, timeout=5.0, throw=True):
        """Perform a Remote Procedure Call (RPC) operation

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param request: None or a Value to qualify this request
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: A Value or Exception

        When invoked with a single name then returns is a single value.
        When invoked with a list of names, then returns a list of values

        >>> ctxt = Context('pva')
        >>> ctxt.put('pv:name', 5.0)
        >>> ctxt.put(['pv:1', 'pv:2'], [1.0, 2.0])
        >>> ctxt.put('pv:name', {'value':5})
        >>>

        The provided value(s) will be automatically coearsed to the target type.
        If this is not possible then an Exception is raised/returned.

        Unless the provided value is a dict, it is assumed to be a plan value
        and an attempt is made to store it in '.value' field.
        """
        done = Queue(maxsize=1)

        ch = self._channel(name)
        op = ch.rpc(done.put_nowait, value, request)
        try:
            result = done.get(timeout=timeout)
            if throw and isinstance(result, Exception):
                raise result

            return result
        except:
            op.cancel()
            raise

    Subscription = Subscription

    def monitor(self, name, cb, request=None):
        """Create a subscription.
        
        :param str name: PV name string
        :param callable cb: Processing callback
        :param request: None or a Value to qualify this request
        :returns: a :py:class:`Subscription` instance
        """
        R = self.Subscription(self, name, cb)
        ch = self._channel(name)

        R._S = ch.monitor(R._event, request)
        try:
            return R
        except:
            S.close()
            raise
