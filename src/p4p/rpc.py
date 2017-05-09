
import logging, inspect
from functools import wraps, partial
_log = logging.getLogger(__name__)

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from .wrapper import Value, Type

def rpc(rtype=None):
    """Decorator marks a proxy method for export.

    :param type: A :py:class:`Type` which the RPC will return

    >>> class Example(object):
        @rpc(NTScalar.buildType('d'))
        def add(self, lhs, rhs):
            return {'value':float(lhs)+flost(rhs)}
    """
    wrap = None
    if isinstance(rtype, Type):
        pass
    elif isinstance(type, (list, tuple)):
        rtype = Type(rtype)
    elif hasattr(rtype, 'type'):
        wrap = rtype.wrap
        rtype = rtype.type
    else:
        raise TypeError("Not supported")

    def wrapper(fn):
        if wrap is not None:
            orig = fn
            @wraps(orig)
            def wrapper(*args, **kws):
                return wrap(orig(*args, **kws))
            fn = wrapper

        fn._reply_Type = rtype
        return fn
    return wrapper

class RemoteError(RuntimeError):
    "Throw with an error message which will be passed back to the caller"

class WorkQueue(object):
    _stopit = object()
    def __init__(self, maxsize=5):
        self._Q = Queue(maxsize=maxsize)
    def push(self, callable):
        self._Q.put_nowait(callable) # throws Queue.Full
    def push_wait(self, callable):
        self._Q.put(callable)
    def interrupt(self):
        self._Q.put(self._stopit)
    def handle(self):
        while True:
            # TODO: Queue.get() (and anything using thread.allocate_lock
            #       ignores signals :(  so timeout periodically to allow delivery
            try:
                callable = self._Q.get(True, 1.0)
            except Empty:
                continue # retry on timeout
            try:
                if callable is self._stopit:
                    break
                callable()
            except:
                _log.exception("Error from WorkQueue")
            finally:
                self._Q.task_done()

class RPCDispatcherBase(object):
    # wrapper to use for request Structures
    Value = Value

    def __init__(self, queue, target=None, channels=set()):
        self.queue = queue
        self.target = target
        self.channels = set(channels)
        M = self.methods = {}
        for name, mem in inspect.getmembers(target):
            if not hasattr(mem, '_reply_Type'):
                continue
            M[name] = mem

    def getMethodNameArgs(self, request):
        raise NotImplementedError("Sub-class must implement getMethodName")
        # sub-class needs to extract method name from request
        # return 'name', {'var':'val'}

    def testChannel(self, name):
        return name in self.channels

    def makeChannel(self, name, src):
        if self.testChannel(name):
            return self # no per-channel tracking needed

    def rpc(self, response, request):
        _log.debug("RPC call %s", request)
        try:
            self.queue.push(partial(self._handle, response, request))
        except Full:
            _log.warn("RPC call queue overflow")
            response.done(error="Too many concurrent RPC calls")

    def _handle(self, response, request):
        try:
            name, args = self.getMethodNameArgs(request)
            fn = self.methods[name]
            rtype = fn._reply_Type

            R = fn(**args)

            if not isinstance(R, Value):
                try:
                    R = self.Value(rtype, R)
                except:
                    _log.exception("Error encoding %s as %s", R, rtype)
                    response.done(error="Error encoding reply")
                    return
            _log.debug("RPC reply %s -> %s", request, R)
            response.done(R)

        except RemoteError as e:
            _log.debug("RPC reply %s -> error: %s", request, e)
            response.done(error=str(e))

        except:
            _log.exception("Error handling RPC %s", request)
            response.done(error="Error handling RPC")
        

class NTURIDispatcher(RPCDispatcherBase):
    """RPC dispatcher using NTURI (a al. eget)

    >>> class Summer(object):
        @rpc([('result', 'i')])
        def add(self, a=None, b=None):
            return {'result': int(a)+int(b)}
    >>> installProvider("arbitrary", NTURIDispatcher(target=Summer(), prefix="pv:prefix:"))

    Making a call with the CLI 'eget' utility::

      $ eget -s pv:prefix:add -a a=1 -a b=2
      ....
      int result 3
    """

    def __init__(self, queue, prefix=None, **kws):
        RPCDispatcherBase.__init__(self, queue, **kws)
        self.prefix = prefix
        self.methods = dict([(prefix+meth, fn) for meth, fn in self.methods.items()])
        self.channels = set(self.methods.keys())
        _log.debug('NTURI methods: %s', ', '.join(self.channels))

    def getMethodNameArgs(self, request):
        # {'schema':'pva', 'path':'pvname', 'query':{'var':'val', ...}}
        return request.path, dict(request.query.tolist())

class MASARDispatcher(RPCDispatcherBase):

    def __init__(self, queue, **kws):
        RPCDispatcherBase.__init__(self, queue, **kws)
        _log.debug("MASAR pv %s methods %s", self.channels, self.methods)

    def getMethodNameArgs(self, request):
        # all through a single PV, method name in request
        # {'function':'rpcname', 'name':['name', ...], 'value':['val', ...]}
        return request.function, dict(zip(request.get('name',[]), request.get('value',[])))
