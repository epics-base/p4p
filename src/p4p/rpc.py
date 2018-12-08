
import logging
import inspect
from functools import wraps, partial
_log = logging.getLogger(__name__)

from threading import Thread

from .wrapper import Value, Type
from .nt import NTURI
from .client.raw import RemoteError, LazyRepr
from .server import DynamicProvider
from .server.raw import SharedPV
from .util import ThreadedWorkQueue, WorkQueue, Full, Empty

__all__ = [
    'rpc',
    'rpccall',
    'rpcproxy',
    'RemoteError',
    'WorkQueue',
    'NTURIDispatcher',
]


def rpc(rtype=None):
    """Decorator marks a method for export.

    :param type: Specifies which :py:class:`Type` this method will return.

    The return type (rtype) must be one of:

    - An instance of :py:class:`p4p.Type`
    - None, in which case the method must return a :py:class:`p4p.Value`
    - One of the NT helper classes (eg :py:class:`p4p.nt.NTScalar`).
    - A list or tuple used to construct a :py:class:`p4p.Type`.

    Exported methods raise an :py:class:`Exception` to indicate an error to the remote caller.
    :py:class:`RemoteError` may be raised to send a specific message describing the error condition.

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
    elif hasattr(rtype, 'type'):  # eg. one of the NT* helper classes
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


def rpccall(pvname, request=None, rtype=None):
    """Decorator marks a client proxy method.

    :param str pvname: The PV name, which will be formated using the 'format' argument of the proxy class constructor.
    :param request: A pvRequest string or :py:class:`p4p.Value` passed to eg. :py:meth:`p4p.client.thread.Context.rpc`.

    The method to be decorated must have all keyword arguments,
    where the keywords are type code strings or :class:`~p4p.Type`.

    """
    def wrapper(fn):
        fn._call_PV = pvname
        fn._call_Request = request
        fn._reply_Type = rtype
        return fn
    return wrapper


class RPCDispatcherBase(DynamicProvider):
    def __init__(self, queue, target=None, channels=set(), name=None):
        DynamicProvider.__init__(self, name, self)  # we are our own Handler
        self.queue = queue
        self.target = target
        self.channels = set(channels)
        self.name = name
        self.__pv = SharedPV(
            handler=self,  # no per-channel state, and only RPC used, so only need on PV
            initial=Value(Type([])),  # we don't support get/put/monitor, so use empty struct
        )
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
        _log.debug("Test RPC channel %s = %s", name, name in self.channels)
        return name in self.channels

    def makeChannel(self, name, src):
        if self.testChannel(name):
            _log.debug("Open RPC channel %s", name)
            return self.__pv  # no per-channel tracking needed
        else:
            _log.warn("Ignore RPC channel %s", name)

    def rpc(self, pv, op):
        _log.debug("RPC call %s", op)
        try:
            self.queue.push(partial(self._handle, op))
        except Full:
            _log.warn("RPC call queue overflow")
            op.done(error="Too many concurrent RPC calls")

    def _handle(self, op):
        try:
            request = op.value()
            name, args = self.getMethodNameArgs(request)
            fn = self.methods[name]
            rtype = fn._reply_Type

            R = fn(**args)

            if not isinstance(R, Value):
                try:
                    R = Value(rtype, R)
                except:
                    _log.exception("Error encoding %s as %s", R, rtype)
                    op.done(error="Error encoding reply")
                    return
            _log.debug("RPC reply %s -> %s", request, LazyRepr(R))
            op.done(R)

        except RemoteError as e:
            _log.debug("RPC reply %s -> error: %s", request, e)
            op.done(error=str(e))

        except:
            _log.exception("Error handling RPC %s", request)
            op.done(error="Error handling RPC")


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
        self.methods = dict([(prefix + meth, fn) for meth, fn in self.methods.items()])
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
        return request.function, dict(zip(request.get('name', []), request.get('value', [])))


def quickRPCServer(provider, prefix, target,
                   maxsize=20,
                   workers=1,
                   useenv=True, conf=None, isolate=False):
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
    :param isolate: Passed to :class:`~p4p.server.Server`
    """
    from p4p.server import Server
    import time
    queue = ThreadedWorkQueue(maxsize=maxsize, workers=workers)
    provider = NTURIDispatcher(queue, target=target, prefix=prefix, name=provider)
    threads = []
    server = Server(providers=[provider], useenv=useenv, conf=conf, isolate=isolate)
    with server, queue:
        while True:
            time.sleep(10.0)


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
    scheme = None  # set to override automatic


def _wrapMethod(K, V):
    pv, req = V._call_PV, V._call_Request
    S = inspect.getargspec(V)

    if S.varargs is not None or S.keywords is not None:
        raise TypeError("vararg not supported for proxy method %s" % K)

    if len(S.args) != len(S.defaults):
        raise TypeError("proxy method %s must specify types for all arguments" % K)

    try:
        NT = NTURI(zip(S.args, S.defaults))
    except Exception as e:
        raise TypeError("%s : failed to build method from %s, %s" % (e, S.args, S.defaults))

    @wraps(V)
    def mcall(self, *args, **kws):
        pvname = pv % self.format
        try:
            uri = NT.wrap(pvname, args, kws, scheme=self.scheme or self.context.name, authority=self.authority)
        except Exception as e:
            raise ValueError("Unable to wrap %s %s as %s (%s)" % (args, kws, NT, e))
        return self.context.rpc(pvname, uri, request=req, timeout=self.timeout, throw=self.throw)

    return mcall


def rpcproxy(spec):
    """Decorator to enable this class to proxy RPC client calls

    The decorated class constructor takes two additional arguments,
    `context=` is required to be a :class:`~p4p.client.thread.Context`.
    `format`= can be a string, tuple, or dictionary and is applied
    to PV name strings given to :py:func:`rpcall`.
    Other arguments are passed to the user class constructor. ::

       @rpcproxy
       class MyProxy(object):
           @rpccall("%s:add")
           def add(lhs='d', rhs='d'):
               pass

       ctxt = Context('pva')
       proxy = MyProxy(context=ctxt, format="tst:")  # evaluates "%s:add"%"tst:"

    The decorated class will be a sub-class of the provided class and :class:`RPCProxyBase`.
    """
    # inject our ctor first so we don't have to worry about super() non-sense.

    def _proxyinit(self, context=None, format={}, **kws):
        assert context is not None, context
        self.context = context
        self.format = format
        spec.__init__(self, **kws)
    obj = {'__init__': _proxyinit}

    for K, V in inspect.getmembers(spec, lambda M: hasattr(M, '_call_PV')):
        obj[K] = _wrapMethod(K, V)

    return type(spec.__name__, (RPCProxyBase, spec), obj)
