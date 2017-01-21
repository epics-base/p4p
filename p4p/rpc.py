
import logging, inspect
_log = logging.getLogger(__name__)

from .wrapper import Value, Type

def rpc(rtype=None):
    if not isinstance(rtype, Type):
        rtype = Type(rtype)
    def wrapper(fn):
        fn._reply_Type = rtype
        return fn
    return wrapper

class RemoteError(RuntimeError):
    pass

class RPCDispatcherBase(object):
    # wrapper to use for request Structures
    Value = Value

    def __init__(self, target=None, channels=set()):
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

    $ eget -s pv:prefix:add -a a=1 -a b=2
    ....
      int result 3
    """

    def __init__(self, prefix=None, **kws):
        RPCDispatcherBase.__init__(self, **kws)
        self.prefix = prefix
        self.methods = dict([(prefix+meth, fn) for meth, fn in self.methods.items()])
        self.channels = set(self.methods.keys())
        _log.debug('NTURI methods: %s', ', '.join(self.channels))

    def getMethodNameArgs(self, request):
        # {'schema':'pva', 'path':'pvname', 'query':{'var':'val', ...}}
        return request.path, dict(request.query.tolist())

class MASARDispatcher(RPCDispatcherBase):

    def __init__(self, **kws):
        RPCDispatcherBase.__init__(self, **kws)
        _log.debug("MASAR pv %s methods %s", self.channels, self.methods)

    def getMethodNameArgs(self, request):
        # all through a single PV, method name in request
        # {'function':'rpcname', 'name':['name', ...], 'value':['val', ...]}
        return request.function, dict(zip(request.name, request.value))
