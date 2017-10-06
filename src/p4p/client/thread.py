
from __future__ import print_function

import logging, warnings, sys
_log = logging.getLogger(__name__)

try:
    from itertools import izip
except ImportError:
    izip = zip
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
from ..nt import _default_wrap, _default_unwrap

__all__ = [
    'Context',
    'Value',
    'Type',
]

if sys.version_info>=(3,0):
    unicode = str

class TimeoutError(RuntimeError):
    def __init__(self):
        RuntimeError.__init__(self, 'Timeout')

class Subscription(object):
    """An active subscription.
    """
    def __init__(self, ctxt, name, cb):
        self._dounwrap = ctxt._dounwrap
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
            if E is not True:
                self._cb(E)
                return
            while True:
                E = self._S.pop()
                if E is None:
                    break
                E = self._dounwrap(E)
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
    """Context(provider, conf=None, useenv=True)

    :param str provider: A Provider name.  Try "pva" or run :py:meth:`Context.providers` for a complete list.
    :param conf dict: Configuration to pass to provider.  Depends on provider selected.
    :param useenv bool: Allow the provider to use configuration from the process environment.
    :param maxsize int: Size of internal work queue used for monitor callbacks
    :param unwrap: Controls :ref:`unwrap`.  Set False to disable

    The methods of this Context will block the calling thread until completion or timeout

    The meaning, and allowed keys, of the configuration dictionary depend on the provider.

    The "pva" provider understands the following keys:

    * EPICS_PVA_ADDR_LIST
    * EPICS_PVA_AUTO_ADDR_LIST
    * EPICS_PVA_SERVER_PORT
    * EPICS_PVA_BROADCAST_PORT
    """
    Value = Value

    name = ''
    "Provider name string"

    providers = raw.Context.providers
    set_debug = raw.Context.set_debug

    def __init__(self, *args, **kws):
        _log.debug("thread.Context with %s %s", args, kws)
        self._Qmax = kws.pop('maxsize', 0)
        unwrap = kws.pop('unwrap', None)
        if unwrap is None:
            self._unwrap = _default_unwrap
        elif not unwrap:
            self._unwrap = {}
        elif isinstance(unwrap, dict):
            self._unwrap = _default_unwrap.copy()
            self._unwrap.update(unwrap)
        else:
            raise ValueError("unwrap must be None, False, or dict, not %s"%unwrap)
        self._ctxt = raw.Context(*args, **kws)
        self.name = self._ctxt.name
        self.disconnect = self._ctxt.disconnect
        self._channel = self._ctxt.channel

        # lazy start threaded WorkQueue
        self._Q, self._T = None, None

    def disconnect(self, name):
        """Drop the named channel from the channel cache.
        The channel will be closed after any pending operations complete.
        """
        pass

    def _dounwrap(self, val):
        if not isinstance(val, Exception):
            fn = self._unwrap.get(val.getID())
            if fn:
                val = fn(val)
        return val

    def _queue(self):
        if self._Q is None:
            Q = WorkQueue(maxsize=self._Qmax)
            T = threading.Thread(name='p4p Context worker', target=Q.handle)
            T.daemon = True
            T.start()
            _log.debug('Started Context worker')
            self._Q, self._T = Q, T
        return self._Q

    def close(self):
        """Force close all Channels and cancel all Operations
        """
        if self._Q is not None:
            _log.debug('Join Context worker')
            self._Q.interrupt()
            self._T.join()
            _log.debug('Joined Context worker')
            self._Q, self._T = None, None
        self._ctxt.close()

    def __del__(self):
        if self._Q is not None:
            warnings.warn("%s collected without close()"%self.__class__)
        self.close()

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def get(self, name, request=None, timeout=5.0, throw=True):
        """Fetch current value of some number of PVs.
        
        :param name: A single name string or list of name strings
        :param request: A :py:class:`p4p.Value` to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.  If False then the Exception is returned instead of the Value

        :returns: A Value or Exception, or list of same

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values

        >>> ctxt = Context('pva')
        >>> V = ctxt.get('pv:name')
        >>> A, B = ctxt.get(['pv:1', 'pv:2'])
        >>>
        """
        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            name = [name]
            if request is not None:
                request = [request]

        if request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)

        # use Queue instead of Event to allow KeyboardInterrupt
        done = Queue(maxsize=len(name))
        result = [TimeoutError()]*len(name)
        ops = [None]*len(name)

        try:
            for i,(N, req) in enumerate(izip(name, request)):
                _log.debug('get %s', N)
                ch = self._channel(N)
                def cb(value, i=i):
                    try:
                        done.put_nowait((value, i))
                    except:
                        _log.exception("Error queuing get result %s", value)
                _log.debug('get %s w/ %s', N, req)
                ops[i] = ch.get(cb, request=req)

            for _n in range(len(name)):
                try:
                    value, i = done.get(timeout=timeout)
                except Empty:
                    if throw:
                        raise TimeoutError()
                    break
                _log.debug('got %s %s', name[i], value)
                if throw and isinstance(value, Exception):
                    raise value
                result[i] = value

        finally:
            [op and op.cancel() for op in ops]

        result = [self._dounwrap(R) for R in result]

        if singlepv:
            return result[0]
        else:
            return result

    def put(self, name, values, request=None, timeout=5.0, throw=True):
        """Write a new value of some number of PVs.
        
        :param name: A single name string or list of name strings
        :param values: A single value or a list of values
        :param request: A :py:class:`p4p.Value` to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: A None or Exception, or list of same

        When invoked with a single name then returns is a single value.
        When invoked with a list of name, then returns a list of values

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
        singlepv = isinstance(name, (bytes, unicode))
        if singlepv:
            name = [name]
            values = [values]
            if request is not None:
                request = [request]

        if request is None:
            request = [None]*len(name)

        assert len(name)==len(request), (name, request)
        assert len(name)==len(values), (name, values)

        # use Queue instead of Event to allow KeyboardInterrupt
        done = Queue(maxsize=len(name))
        result = [TimeoutError()]*len(name)
        ops = [None]*len(name)

        try:
            for i,(n, value, req) in enumerate(izip(name, values, request)):
                if isinstance(value, (bytes, unicode)) and value[:1]=='{':
                    try:
                        value = json.loads(value)
                    except ValueError:
                        raise ValueError("Unable to interpret '%s' as json"%value)

                ch = self._channel(n)

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

            for _n in range(len(name)):
                try:
                    value, i = done.get(timeout=timeout)
                except Empty:
                    if throw:
                        raise TimeoutError()
                    break
                if throw and isinstance(value, Exception):
                    raise value
                result[i] = value

            if singlepv:
                return result[0]
            else:
                return result
        finally:
            [op and op.cancel() for op in ops]

    def rpc(self, name, value, request=None, timeout=5.0, throw=True):
        """Perform a Remote Procedure Call (RPC) operation

        :param str name: PV name string
        :param Value value: Arguments.  Must be Value instance
        :param request: A :py:class:`p4p.Value` to qualify this request, or None to use a default.
        :param float timeout: Operation timeout in seconds
        :param bool throw: When true, operation error throws an exception.
                     If False then the Exception is returned instead of the Value

        :returns: A Value or Exception

        When invoked with a single name then returns is a single value.
        When invoked with a list of names, then returns a list of values

        >>> ctxt = Context('pva')
        >>> ctxt.rpc('pv:name:add', {'A':5, 'B'; 6})
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
            try:
                result = done.get(timeout=timeout)
            except Empty:
                result = TimeoutError()
            if throw and isinstance(result, Exception):
                raise result

            return self._dounwrap(result)
        except:
            op.cancel()
            raise

    Subscription = Subscription

    def monitor(self, name, cb, request=None):
        """Create a subscription.
        
        :param str name: PV name string
        :param callable cb: Processing callback
        :param request: A :py:class:`p4p.Value` to qualify this request, or None to use a default.
        :returns: a :py:class:`Subscription` instance

        The callable will be invoked with one argument which is either.

        * A Value
        * A sub-class of Exception
        * None when the subscription is complete, and more update will ever arrive.
        """
        R = self.Subscription(self, name, cb)
        ch = self._channel(name)

        R._S = ch.monitor(R._event, request)
        try:
            return R
        except:
            S.close()
            raise
