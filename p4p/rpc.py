
import logging
_log = logging.getLogger(__name__)

from .wrapper import Value, Type

def rpc(rtype):
    if not isinstance(rtype, Type):
        rtype = Type(rtype)
    def wrapper(fn):
        fn._reply_Type = rtype
        return fn
    return wrapper

class RemoteError(RuntimeError):
    pass

class URIDispatcher(object):
    """RPC dispatcher using NTURI (a al. eget)

    >>> class Example(URIDispatcher):
        @rpc([('field', 'i')])
        def add(self, a=None, b=None):
            return {'field': int(a)+int(b)}
    >>> installProvider("providername", Example("pv:prefix:"))

    $ eget -s foo:add -a a=1 -a b=2
    ....
      int result 3
    """

    # wrapper to use for request Structures
    Value = Value
    
    def __init__(self, prefix):
        self.prefix = prefix

    def testChannel(self, name):
        if not name.startswith(self.prefix):
            return False
        name = name[len(self.prefix):]
        fn = getattr(self, name, None)
        rtype = fn and getattr(fn, '_reply_Type', None)
        return rtype is not None

    def makeChannel(self, name, src):
        if self.testChannel(name):
            return self # no per-channel tracking needed

    def rpc(self, response, request):
        # {'schema':'pva', 'path':'pvname', 'query':{'name':'value', ...}}
        _log.debug("RPC call %s", request)

        try:
            fn = getattr(self, request.path[len(self.prefix):])
            rtype = fn._reply_Type # must be decorated
            args = dict(request.query.tolist())
            R = fn(**args)
            _log.debug("RPC reply %s -> %s", request, R)
            response.done(Value(rtype, R))
        except RemoteError as e:
            _log.debug("RPC reply %s -> error: %s", request, e)
            response.done(error=str(e))
        except:
            _log.exception("Error handling RPC %s", request)
            response.done(error="Error handling RPC")

class MASARDispatcher(object):
    Value = Value
    
    def __init__(self, pv):
        self.pv = pv

    def testChannel(self, name):
        return self.pv==name

    def makeChannel(self, name, src):
        if self.pv==name:
            return self

    def rpc(self, response, request):
        # {'function':'rpcname', 'names':['name', ...], 'values':['val', ...]}
        _log.debug("RPC call %s", request)

        try:
            fn = getattr(self, request.function)
            rtype = fn._reply_Type
            args = dict(zip(request.names, request.values))
            R = fn(**args)
            _log.debug("RPC reply %s -> %s", request, R)
            response.done(Value(rtype, R))
        except RemoteError as e:
            _log.debug("RPC reply %s -> error: %s", request, e)
            response.done(error=str(e))
        except:
            _log.exception("Error handling RPC %s", request)
            response.done(error="Error handling RPC")
