
import logging, inspect
from functools import wraps, partial
_log = logging.getLogger(__name__)

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty
from threading import Thread

__all__ = [
    'rpc',
    'rpccall',
    'rpcproxy',
    'RemoteError',
    'WorkQueue',
    'NTURIDispatcher',
]

from .wrapper import Value, Type
from .nt import NTURI

def rpc(rtype=None):
    """Decorator marks a proxy method for export.

    :param type: A :py:class:`Type` which the RPC will return

    >>> class Example(object):
        @rpc(NTScalar.buildType('d'))
        def add(self, lhs, rhs):
            return {'value':float(lhs)+flost(rhs)}
    """
    wrap = None
    if rtype is None or isinstance(rtype, Type):
        pass
    elif isinstance(type, (list, tuple)):
        rtype = Type(rtype)
    elif hasattr(rtype, 'type'): # eg. one of the NT* helper classes
        wrap = rtype.wrap
        rtype = rtype.type
    else:
        raise TypeError("Not supported")

    def wrapper(fn):
        if wrap is not None:
            orig = fn
            @wraps(orig)
            def wrapper2(*args, **kws):
                return wrap(orig(*args, **kws))
            fn = wrapper2

        fn._reply_Type = rtype
        return fn
    return wrapper

def rpccall(pvname, rtype=None, request=None):
    """Decorator marks a client proxy method.
    
    The method to be decorated must have all keyword arguments,
    where the keywords are type code strings or :class:`~p4p.Type`.
    """
    def wrapper(fn):
        fn._call_PV = pvname
        fn._call_Request = request
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
        """Break one call to handle()

        eg. Call N times to break N threads.

        This call blocks if the queue is full.
        """
        self._Q.put(self._stopit)
    def handle(self):
        """Process queued work until interrupt() is called
        """
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

    Method names are prefixed with a fixed string.

    >>> queue = WorkQueue()
    >>> class Summer(object):
        @rpc([('result', 'i')])
        def add(self, a=None, b=None):
            return {'result': int(a)+int(b)}
    >>> installProvider("arbitrary", NTURIDispatcher(queue, target=Summer(), prefix="pv:prefix:"))

    Making a call with the CLI 'eget' utility::

      $ eget -s pv:prefix:add -a a=1 -a b=2
      ....
      int result 3

    :param queue WorkQueue: A WorkQueue to which RPC calls will be added
    :param prefix str: PV name prefix used by RPC methods
    :param target: The object which has the RPC calls
    """

    def __init__(self, queue, prefix=None, **kws):
        RPCDispatcherBase.__init__(self, queue, **kws)
        self.prefix = prefix
        self.methods = dict([(prefix+meth, fn) for meth, fn in self.methods.items()])
        self.channels = set(self.methods.keys())
        _log.debug('NTURI methods: %s', ', '.join(self.channels))

    def getMethodNameArgs(self, request):
        # {'schema':'pva', 'path':'pvname', 'query':{'var':'val', ...}}
        return request.path, dict(request.query.items())

# legecy for MASAR only
# do not use in new code
class MASARDispatcher(RPCDispatcherBase):

    def __init__(self, queue, **kws):
        RPCDispatcherBase.__init__(self, queue, **kws)
        _log.debug("MASAR pv %s methods %s", self.channels, self.methods)

    def getMethodNameArgs(self, request):
        # all through a single PV, method name in request
        # {'function':'rpcname', 'name':['name', ...], 'value':['val', ...]}
        return request.function, dict(zip(request.get('name',[]), request.get('value',[])))

def quickRPCServer(provider, prefix, target,
                   maxsize=20,
                   workers=1,
                   useenv=True, conf=None):
    """Run an RPC server in the current thread

    Calls are handled sequentially, and always in the current thread, if workers=1 (the default).
    If workers>1 then calls are handled concurrently by a pool of worker threads.
    Requires NTURI style argument encoding.

    :param str provider: A provider name.  Must be unique in this process.
    :param str prefix: PV name prefix.  Along with method names, must be globally unique.
    :param target: The object which is exporting methods.  (use the :func:`rpc` decorator)
    :param int maxsize: Number of pending RPC calls to be queued.
    :param int workers: Number of worker threads (default 1)
    :param useenv: Passed to :class:`~p4p.server.Server`
    :param conf: Passed to :class:`~p4p.server.Server`
    """
    from p4p.server import Server, installProvider, removeProvider
    queue = WorkQueue(maxsize=maxsize)
    installProvider(provider, NTURIDispatcher(queue, target=target, prefix=prefix))
    try:
        threads = []
        server = Server(providers=provider, useenv=useenv, conf=conf)
        try:
            for n in range(1,workers):
                T = Thread(name='%s Worker %d'%(provider, n), target=queue.handle)
                threads.append(T)
                T.start()
            # handle calls in the current thread until KeyboardInterrupt
            queue.handle()
        finally:
            try:
                for T in threads:
                    queue.interrupt()
                    T.join()
            finally:
                # we really need to do this or the process will hang on exit
                server.stop()
    finally:
        removeProvider(provider)

class RPCProxyBase(object):
    """Base class for automatically generated proxy classes
    """
    context = None
    "The Context provided on construction"
    format = None
    "The tuple/dict used to format ('%' operator) PV name strings."
    timeout = 3.0
    "Timeout of RPC calls in seconds"
    authority = ''
    "Authority string sent with NTURI requests"
    throw = True
    "Whether call errors raise an exception, or return it"
    scheme = None # set to override automatic

def _wrapMethod(K, V):
    pv, req = V._call_PV, V._call_Request
    S = inspect.getargspec(V)

    if S.varargs is not None or S.keywords is not None:
        raise TypeError("vararg not supported for proxy method %s"%K)

    if len(S.args)!=len(S.defaults):
        raise TypeError("proxy method %s must specify types for all arguments"%K)

    NT = NTURI(zip(S.args, S.defaults))

    @wraps(V)
    def mcall(self, *args, **kws):
        pvname = pv%self.format
        pos = dict(zip(S.args[:len(args)], args))
        pos.update(kws)
        uri = NT.wrap(pvname, pos, scheme=self.scheme or self.context.name, authority=self.authority)
        return self.context.rpc(pvname, uri, request=req, timeout=self.timeout, throw=self.throw)

    return mcall

def rpcproxy(spec):
    """Decorator to enable this class to proxy RPC client calls
    
    The decorator class constructor takes one additional arugment "context"
    which should by a :class:`~p4p.client.thread.Context`. ::
    
       @rpcproxy
       class MyProxy(object):
           @rpccall("%s:add")
           def add(lhs='d', rhs='d'):
               pass

    The decorated class will by a sub-class of the provided class and :class:`RPCProxyBase`.
    """
    # inject our ctor first so we don't have to worry about super() non-sense.
    def _proxyinit(self, context=None, format={}, **kws):
        assert context is not None, context
        self.context = context
        self.format = format
        spec.__init__(self, **kws)
    obj = {'__init__':_proxyinit}
        
    for K,V in inspect.getmembers(spec, lambda M:hasattr(M, '_call_PV')):
        obj[K] = _wrapMethod(K, V)

    return type(spec.__name__, (RPCProxyBase, spec), obj)
