
import logging
_log = logging.getLogger(__name__)

from itertools import izip
import json

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from . import raw
from ..wrapper import Value

class Context(object):
    Value = Value
    def __init__(self, *args, **kws):
        self._ctxt = raw.Context(*args, **kws)

        self._channels = {}

    def close(self):
        self._channels = None
        self._ctxt.close()

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
                    print 'foo', type
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
                    print 'bar', type
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
