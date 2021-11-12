# distutils: language = c++
#cython: language_level=2

cimport cython
from libc.stdint cimport uint64_t, int64_t
from libc.string cimport strcpy
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map as cmap
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from cpython cimport PyObject
from cpython.bytes cimport PyBytes_FromStringAndSize

cimport numpy

from pvxs cimport version as _version
from pvxs cimport log
from pvxs cimport util
from pvxs cimport sharedArray
from pvxs cimport data
from pvxs cimport nt
from pvxs cimport client
from pvxs cimport server
from pvxs cimport source
from pvxs cimport sharedpv

numpy.import_array()
log.logger_config_env()

if not _version.version_abi_check():
    raise RuntimeError("PVXS ABI mismatch")

cdef extern from "<p4p.h>" namespace "p4p":
    # p4p.h redefines/overrides some definitions from Python.h (eg. PyMODINIT_FUNC)

    # pvxs_type.cpp
    data.TypeDef startPrototype(const string& id, const data.Value& base) except+
    void appendPrototype(data.TypeDef&, object spec) except+
    object asPySpec(const data.Value& v) except+

    # pvxs_value.cpp
    int except_map()
    object asPy(const data.Value& v, bool unpackstruct, bool unpackrecurse, object wrapper) except+
    void storePy(data.Value& v, object py) except+ except_map
    object tostr(const data.Value& v, size_t limit, bool showval) except+

    # pvxs_sharedpv.cpp
    string toString(const server.Server& serv, int detail) nogil except+
    void attachHandler(sharedpv.SharedPV& pv, object handler) except+
    void detachHandler(sharedpv.SharedPV& pv) except+

    # pvxs_source.cpp
    shared_ptr[server.Source] createDynamic(object) except+
    void disconnectDynamic(const shared_ptr[server.Source]& src) except+

    # pvxs_client.cpp
    void opHandler[Builder](Builder& builder, object handler)
    void opBuilder[Builder](Builder& builder, object handler)
    void opEvent(client.MonitorBuilder& builder, object handler)
    object monPop(const shared_ptr[client.Subscription]& mon) with gil

############### version

def version():
    return _version.version_int()

def version_str():
    return _version.version_str().decode('utf-8')

############### log

logLevelAll = <int>log.Debug
logLevelTrace = <int>log.Debug
logLevelDebug = <int>log.Debug
logLevelInfo = <int>log.Info
logLevelWarn = <int>log.Warn
logLevelError = <int>log.Err
logLevelFatal = <int>log.Crit
logLevelOff = <int>log.Crit

def logger_level_set(basestring name, int lvl):
    log.logger_level_set(name.encode(), lvl)

############### util

def listRefs():
    ret = {}
    for P in util.instanceSnapshot():
        ret[P.first.decode('utf-8')] = P.second
    return ret

def _forceLazy():
    pass

############### data

# py object to hold ownership
cdef class SharedArray:
    cdef sharedArray.shared_array[const void] arr

cdef public:
    cdef pvxs_wrap_array(const sharedArray.shared_array[const void]& arr):
        cdef SharedArray wrap = SharedArray.__new__(SharedArray)
        wrap.arr = arr
        return wrap

cdef lookupMember(data.Value* dest, const data.Value& top, key, int err):
        cdef string ckey
        cdef data.Value mem

        if key is None:
            dest[0] = top
            return
        elif isinstance(key, unicode):
            ckey = key.encode()
        else:
            ckey = key

        try:
            dest[0] = top.lookup(ckey)
        except KeyError as e:
            if err==0:
                return
            elif err==1:
                raise
            else:
                raise AttributeError(key)

cdef class _Value:
    cdef data.Value val

    def __init__(self, type=None, value=None, clone=None):
        cdef _Type base
        cdef _Value val

        if self.val.valid():
            pass

        elif type is not None:
            base = type
            self.val = base.proto.cloneEmpty()
            if value is not None:
                storePy(self.val, value)

        elif clone is not None:
            val = clone
            self.val = val.val.clone()

        else:
            raise ValueError("type= or clone= required")

    def get(self, key, default=None):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 0)
        if not mem.valid():
            return default

        return asPy(mem, False, False, None)

    def items(self, key=None):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 1)

        return asPy(mem, True, False, None)

    def __setitem__(self, key, value):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 1)

        storePy(mem, value)

    def __getitem__(self, key):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 1)

        return asPy(mem, False, False, None)

    def __setitem__(self, key, value):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 1)

        storePy(mem, value)

    def __getattr__(self, key):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 2)

        return asPy(mem, False, False, None)

    def __setattr__(self, key, value):
        cdef data.Value mem
        lookupMember(&mem, self.val, key, 2)

        storePy(mem, value)

    def has(self, basestring name):
        cdef string cname = name.encode()
        return self.val[cname].valid()

    def tolist(self, name=None):
        cdef data.Value mem
        lookupMember(&mem, self.val, name, 2)

        return asPy(mem, True, True, None)

    def todict(self, name=None, wrapper=dict):
        cdef data.Value mem
        lookupMember(&mem, self.val, name, 2)

        return asPy(mem, True, True, wrapper)

    def getID(self):
        return self.val.id().decode() or 'structure'

    def __iter__(self):
        ret = []
        for child in self.val.ichildren():
            ret.append(self.val.nameOf(child))
        return iter(ret)

    def type(self, fld=None):
        cdef data.Value mem
        cdef _Type type
        lookupMember(&mem, self.val, fld, 1)

        type = Type.__new__(Type)
        type.proto = mem.cloneEmpty()
        return type

    def select(self, basestring name, basestring selector=None):
        cdef string cname = name
        cdef data.Value u

        if self.val.type().code!=data.Union:
            raise KeyError(name)

        # selection is a side-effect of union lookup
        if selector is not None:
            u = self.val[cname]
        else:
            u._from(NULL);

    def changed(self, field=None):
        cdef data.Value mem
        lookupMember(&mem, self.val, field, 1)

        return mem.isMarked(True, True)

    def changedSet(self, expand=False, parents=False):
        cdef bool exp = expand
        cdef bool par = parents
        cdef data.Value parent

        ret = set()

        for mem in self.val.imarked():
            ret.add(self.val.nameOf(mem).decode())

            if exp and mem.type().code==data.Struct:
                for smem in mem.iall():
                    ret.add(self.val.nameOf(smem).decode())

            if par:
                parent = mem[b"<"]
                while parent.valid() and not self.val.equalInst(parent):
                    ret.add(self.val.nameOf(parent).decode())
                    parent = parent[b"<"]

        return ret

    def mark(self, field, val=True):
        cdef data.Value mem
        lookupMember(&mem, self.val, field, 1)

        mem.mark(val)

    def unmark(self):
        self.val.unmark(True, True)

    def tostr(self, int limit=0):
        return tostr(self.val, limit, True)
        

cdef public api:
    cdef bool pvxs_isValue(object v):
        return isinstance(v, _Value)

    cdef data.Value pvxs_extract(object v):
        if isinstance(v, _Value):
            return (<_Value>v).val
        elif isinstance(v, _Type):
            return (<_Type?>v).proto
        else:
            return data.Value()

    cdef object pvxs_pack(const data.Value& v):
        cdef _Value raw = Value.__new__(Value)
        raw.val = v
        return raw

cdef class _Type:
    cdef data.Value proto

    def __init__(self, spec, basestring id=None, base=None):
        cdef data.TypeDef tdef
        cdef data.Value cbase
        cdef string cid

        if id is not None and base is not None:
            raise ValueError("only pass one of id= or base=")
        if id is not None:
            cid = id.encode()

        if base is not None:
            cbase = (<_Type?>base).proto

        tdef = startPrototype(cid, cbase)
        appendPrototype(tdef, spec)
        self.proto = tdef.create()

    def getID(self):
        return self.proto.id().decode() or 'structure'

    def keys(self):
        ret = []
        for child in self.proto.ichildren():
            ret.append(self.proto.nameOf(child).decode())
        return ret

    def aspy(self, basestring name=None):
        cdef string cname
        if name is None:
            return asPySpec(self.proto)
        else:
            cname = name.encode()
            return asPySpec(self.proto.lookup(cname))

    def has(self, basestring name):
        cdef string cname = name.encode()
        return self.proto[cname].valid()

    def __getitem__(self, name):
        cdef string cname = name.encode()
        cdef data.Value fld = self.proto.lookup(cname)
        cdef _Type ret
        if not fld.valid():
            raise KeyError(name)
        elif fld.type().code==data.Struct:
            ret = Type.__new__(Type)
            ret.proto = fld
            return ret
        else:
            return asPySpec(fld)

    def __len__(self):
        return self.proto.nmembers()

    def tostr(self, int limit=0):
        return tostr(self.proto, limit, False)

cdef public:
    cdef bool pvxs_isType(object v):
        return isinstance(v, _Type)

# sub-class hooks
Value = _Value
Type = _Type

############### internal

cdef splitArgs(vector[string]& out, basestring s):
    for v in s.split(' '):
        out.push_back(v.encode())

############### client

class Cancelled(RuntimeError):
    "Cancelled from client end."
    def __init__(self, msg=None):
        RuntimeError.__init__(self, msg or "Cancelled by client")

class Disconnected(RuntimeError):
    "Channel becomes disconected."
    def __init__(self, msg=None):
        RuntimeError.__init__(self, msg or "Channel disconnected")

class Finished(Disconnected):
    "Special case of Disconnected when a Subscription has received all updates it will ever receive."
    def __init__(self, msg=None):
        Disconnected.__init__(self, msg or "Subscription Finished")

class RemoteError(RuntimeError):
    "Thrown with an error message which has been sent by a server to its remote client"

cdef public:
    cdef object _Cancelled = Cancelled
    cdef object _Disconnected = Disconnected
    cdef object _Finished = Finished
    cdef object _RemoteError = RemoteError

# can't tp_clear as we have no way to replace Operation handler (cancel?)
@cython.no_gc_clear
cdef class ClientOperation:
    cdef shared_ptr[client.Operation] op
    cdef object handler
    cdef object builder
    cdef object __weakref__

    def __init__(self, ClientProvider ctxt, basestring name,
                 handler=None, _Value value=None, builder=None, _Value pvRequest=None, get=False,
                 put=False, rpc=False):
        cdef string pvname = name.encode()
        cdef bool doGet = get
        cdef bool doPut = put
        cdef bool doRPC = rpc
        cdef client.GetBuilder bget
        cdef client.PutBuilder bput
        cdef client.RPCBuilder brpc

        self.handler = handler
        self.builder = builder

        if doGet and not doPut and not doRPC:
            bget = ctxt.ctxt.get(pvname)
            opHandler(bget, handler)
            if pvRequest is not None:
                bget.rawRequest(pvRequest.val)
            with nogil:
                self.op = bget.exec_()

        elif doPut and not doRPC:
            bput = ctxt.ctxt.put(pvname).fetchPresent(doGet)
            opHandler(bput, handler)
            opBuilder(bput, builder)
            if pvRequest is not None:
                bput.rawRequest(pvRequest.val)
            with nogil:
                self.op = bput.exec_()

        elif not doGet and not doPut and doRPC:
            brpc = ctxt.ctxt.rpc(pvname, value.val)
            opHandler(brpc, handler)
            if pvRequest is not None:
                brpc.rawRequest(pvRequest.val)
            with nogil:
                self.op = brpc.exec_()

        else:
            raise ValueError("Operation unsupported combination of get=, put=, and rpc=")

    def __dealloc__(self):
        self._close()

    def close(self):
        self._close()

    cdef _close(self):
        cdef shared_ptr[client.Operation] op
        cdef bool cancelled = False
        self.op.swap(op)
        if <bool>op:
            with nogil:
                cancelled = op.get().cancel()
                op.reset()
        if cancelled:
            self.handler(1, "", None)

# can't tp_clear as we have no way to replace Subscription handler (cancel?)
@cython.no_gc_clear
cdef class ClientMonitor:
    cdef shared_ptr[client.Subscription] sub
    cdef object handler
    cdef object __weakref__

    def __init__(self, ClientProvider ctxt, basestring name, handler=None, _Value pvRequest=None):
        cdef string pvname = name.encode()
        cdef client.MonitorBuilder builder

        self.handler = handler

        builder = ctxt.ctxt.monitor(pvname) \
                      .maskConnected(True) \
                      .maskDisconnected(False)
        opEvent(builder, handler)
        if pvRequest is not None:
            builder.rawRequest(pvRequest.val)
        self.sub = builder.exec_()

    def __dealloc__(self):
        self._close()

    def close(self):
        self._close()

    cdef _close(self):
        cdef shared_ptr[client.Subscription] trash
        self.sub.swap(trash)
        if <bool>trash:
            with nogil:
                trash.get().cancel()

    def pop(self):
        cdef shared_ptr[client.Subscription] sub = self.sub # local copy to guard against concurrent _close()
        if <bool>sub:
            return monPop(sub) # will unlock/relock GIL

cdef class ClientProvider:
    def __init__(self, basestring provider, conf=None, useenv=True):
        cdef client.Config cconf
        cdef server.Config_defs_t defs

        if useenv:
            cconf.applyEnv()

        if conf is not None:
            for K,V in conf.items():
                defs[K.encode()] = V.encode()

            cconf.applyDefs(defs)

        with nogil:
            self.ctxt = cconf.build()

    def __dealloc__(self):
        with nogil:
            self.ctxt = client.Context()

    def conf(self):
        cdef client.Config_defs_t defs
        ret = {}
        self.ctxt.config().updateDefs(defs)
        for K,V in defs:
            ret[K.decode()] = V.decode()
        return ret

    def hurryUp(self):
        with nogil:
            self.ctxt.hurryUp()

    def close(self):
        with nogil:
            self.ctxt = client.Context()

    def disconnect(self, basestring name=None):
        cdef string cname
        if name is not None:
            cname = name
        with nogil:
            self.ctxt.cacheClear(cname, client.cacheAction.Disconnect)

    @staticmethod
    def makeRequest(basestring desc):
        cdef _Value ret = Value.__new__(Value)

        ret.val = client.Context.request().pvRequest(desc.encode()).build()

        return ret

############### server

# global provider registry
_providers = {}

cdef class Server:
    def __init__(self, conf=None, useenv=True, providers=None):
        cdef server.Config sconf
        cdef server.Config_defs_t defs
        cdef StaticProvider sprov
        cdef DynamicProvider dprov
        cdef Source src
        cdef int iorder

        if useenv:
            sconf.applyEnv()

        if conf is not None:
            for K,V in conf.items():
                defs[K.encode()] = V.encode()

            sconf.applyDefs(defs)

        with nogil:
            self.serv = sconf.build()

        for prov, order in providers:
            iorder = order
            if prov in _providers:
                prov = _providers[prov]() # lock weakref

            if isinstance(prov, StaticProvider):
                sprov = prov
                with nogil:
                    self.serv.addSource(sprov.name, sprov.src.source(), iorder)

            elif isinstance(prov, DynamicProvider):
                dprov = prov
                with nogil:
                    self.serv.addSource(dprov.name, dprov.src, iorder)

            elif isinstance(prov, Source):
                src = prov
                with nogil:
                    self.serv.addSource(src.name, src.src, iorder)

            else:
                raise ValueError("Unsupported provider type %s"%type(prov))

    def __dealloc__(self):
        with nogil:
            self.serv = server.Server()

    def run(self):
        with nogil:
            self.serv.run()

    def interrupt(self):
        with nogil:
            self.serv.interrupt()

    def start(self):
        with nogil:
            self.serv.start()

    def stop(self):
        with nogil:
            self.serv.stop()

    def conf(self):
        cdef server.Config_defs_t defs
        ret = {}

        self.serv.config().updateDefs(defs)
        self.serv.clientConfig().updateDefs(defs)

        for K,V in defs:
            ret[K.decode()] = V.decode()
        return ret

    @property
    def guid(self):
        cdef server.ServerGUID guid = self.serv.config().guid
        return PyBytes_FromStringAndSize(<char*>guid.data(), guid.size())

    def tostr(self, int detail=0):
        return toString(self.serv, detail).decode()

# Avoid need for custom tp_clear to detachHandler() prior to CLEAR(handler).
@cython.no_gc_clear
cdef class SharedPV:
    cdef sharedpv.SharedPV pv
    cdef object handler
    cdef object __weakref__

    def __init__(self, handler=None, options=None):
        self.pv = sharedpv.SharedPV.buildReadonly()
        if handler is not None:
            attachHandler(self.pv, handler)
        self.handler = handler

    def __dealloc__(self):
        detachHandler(self.pv)
        with nogil:
            self.pv = sharedpv.SharedPV()

    def open(self, _Value value):
        with nogil:
            self.pv.open(value.val)

    def post(self, _Value value):
        cdef data.Value temp
        with nogil:
            self.pv.post(value.val)

    def current(self):
        cdef _Value ret = Value.__new__(Value)
        with nogil:
            ret.val = self.pv.fetch()
        return ret

    def close(self, destroy=False):
        with nogil:
            self.pv.close()

    def isOpen(self):
        cdef bool ret
        with nogil:
            ret = self.pv.isOpen()
        return ret

cdef public:
    cdef sharedpv.SharedPV SharedPV_unwrap(object pv):
        cdef sharedpv.SharedPV ret
        if isinstance(pv, SharedPV):
            ret = (<SharedPV>pv).pv
        return ret

cdef class ServerOperation:
    """An in-progress Put or RPC operation from a client.
    """
    cdef shared_ptr[source.ExecOp] op
    cdef data.Value val

    def __dealloc__(self):
        with nogil:
            self.op.reset()

    def pvRequest(self):
        '''pvRequest() -> Value
        Access the request Value provided by the client, which may ignored, or used to modify handling.
        '''
        cdef _Value ret = Value.__new__(Value)
        ret.val = self.op.get().pvRequest()
        return ret

    def value(self):
        '''value() -> Value
        For an RPC operation, the argument Value provided
        '''
        cdef _Value ret = Value.__new__(Value)
        ret.val = self.val
        return ret

    def name(self):
        '''name() -> str
        The PV name used by the client
        '''
        return self.op.get().name().decode()

    def peer(self):
        '''peer() -> str
        Client address
        '''
        return self.op.get().peerName().decode()

    def account(self):
        '''account() -> str
        Client identity
        '''
        return self.op.get().credentials().get().account.decode()

    def roles(self):
        '''roles() -> {str}
        Client group memberships
        '''
        ret = set()
        for role in self.op.get().credentials().get().roles():
            ret.add(role.decode())
        return ret

    def done(self, _Value value=None, basestring error=None):
        '''done(value=None, error=None)

        Signal completion of the operation. ::

          # successful completion without result (Put or RPC)
          done()
          # successful completion with result (RPC only)
          done(Value)
          # unsuccessful completion (Put or RPC)
          done(error="msg")
        '''
        cdef string msg

        if error is not None:
            msg = error.encode()
            with nogil:
                self.op.get().error(msg)

        elif value is None:
            with nogil:
                self.op.get().reply()

        else:
            with nogil:
                self.op.get().reply(value.val)

    def info(self, msg):
        pass
    def warn(self, msg):
        pass

cdef public:
    cdef object ServerOperation_wrap(const shared_ptr[source.ExecOp]& op, const data.Value& val):
        cdef ServerOperation ret = ServerOperation.__new__(ServerOperation)
        ret.op = op;
        ret.val = val
        return ret

cdef class StaticProvider:
    cdef string name
    cdef sharedpv.StaticSource src
    cdef object shadow
    cdef object __weakref__

    def __init__(self, basestring name):
        self.name = name.encode()
        self.src = sharedpv.StaticSource.build()
        self.shadow = {}

    def __dealloc__(self):
        with nogil:
            self.src = sharedpv.StaticSource()

    def close(self):
        with nogil:
            self.src.close()

    def add(self, basestring name, SharedPV pv):
        cdef string cname = name.encode()
        with nogil:
            self.src.add(cname, pv.pv)
        self.shadow[name] = pv

    def remove(self, name):
        cdef string cname = name.encode()
        self.shadow.pop(name, None)
        with nogil:
            self.src.remove(cname)

    def keys(self):
        cdef cmap[string, server.SharedPV] pvs
        with nogil:
            pvs = self.src.list()
        ret = []
        for pair in pvs:
            ret.append(pair.first.decode())
        return ret

# Avoid need for custom tp_clear to disconnectDynamic() prior to CLEAR(handler).
@cython.no_gc_clear
cdef class DynamicProvider:
    cdef string name
    cdef shared_ptr[source.Source] src
    cdef object handler
    cdef object __weakref__

    def __init__(self, basestring name, handler):
        self.name = name.encode()
        self.handler = handler
        self.src = createDynamic(handler)

    def __dealloc__(self):
        disconnectDynamic(self.src)

cdef class Source:
    pass
