# distutils: language = c++
#cython: language_level=2

cimport cython

from libc.stddef cimport size_t
from libc.string cimport strchr
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.set cimport set
from libcpp.vector cimport vector

from cpython.object cimport PyObject, PyTypeObject, traverseproc, visitproc
from cpython.ref cimport Py_INCREF, Py_XDECREF

cdef extern from "<osiSock.h>" nogil:
    ctypedef int SOCKET
    enum:INVALID_SOCKET
    struct in_addr:
        unsigned long s_addr
    struct sockaddr:
        pass
    struct sockaddr_in:
        int sin_family
        unsigned short sin_port
        in_addr sin_addr
    union osiSockAddr:
        sockaddr sa
        sockaddr_in ia
    SOCKET epicsSocketCreate ( int domain, int type, int protocol )
    void epicsSocketDestroy(SOCKET sock)

    unsigned sockAddrToDottedIP(const sockaddr* paddr, char* pBuf, unsigned bufSize )
    int aToIPAddr(const char* pAddrString, unsigned short defaultPort, sockaddr_in* pIP)

    enum: AF_INET

cdef extern from "<pv/discoverInterfaces.h>" namespace "epics::pvAccess" nogil:
    cdef struct ifaceNode:
        osiSockAddr addr
        osiSockAddr peer
        osiSockAddr bcast
        osiSockAddr mask
        bool loopback
        bool validP2P
        bool validBcast

    int discoverInterfaces(vector[ifaceNode]& list, SOCKET socket, const osiSockAddr *pMatchAddr) except+

cdef showINet(const osiSockAddr& addr):
    cdef char buf[30]
    cdef char *sep
    sockAddrToDottedIP(&addr.sa, buf, sizeof(buf))
    sep = strchr(buf, ':')
    if sep:
        sep[0] = '\0'
    return (<bytes>buf).decode('UTF-8')

cdef class IFInfo(object):
    @staticmethod
    def current(int domain, int type, unicode match=None):
        cdef SOCKET sock
        cdef vector[ifaceNode] info
        cdef osiSockAddr maddr
        cdef osiSockAddr *pmatch = NULL

        if match is not None:
            if aToIPAddr(match.encode('UTF-8'), 0, &maddr.ia):
                raise ValueError("%s not an IP address"%match)
            pmatch = &maddr

        sock = epicsSocketCreate(domain, type, 0)
        if sock==INVALID_SOCKET:
            raise RuntimeError("Unable to allocate socket")
        try:
            if discoverInterfaces(info, sock, pmatch):
                raise RuntimeError("Unable to inspect network interfaces")
        finally:
            epicsSocketDestroy(sock)

        ret = []
        for node in info:
            if node.addr.ia.sin_family!=AF_INET:
                continue
            ent = {
                'loopback':node.loopback,
                'addr':showINet(node.addr),
            }
            if node.mask.ia.sin_family==AF_INET:
                ent['mask'] = showINet(node.mask)
            if node.validP2P:
                ent['peer'] = showINet(node.peer)
            if node.validBcast:
                ent['bcast'] = showINet(node.bcast)
            ret.append(ent)

        return ret

## PVD explicitly uses std::tr1::shared_ptr<> which is _sometimes_ a
## distinct type, and sometimes an alias for std::shared_ptr<>.
## So instead of:
#from libcpp.memory cimport shared_ptr, weak_ptr
## We replicate a minimal (noexcept) subset of memory.pxd for std::tr1::shared_ptr
cdef extern from "<pv/sharedPtr.h>" namespace "std::tr1" nogil:
    cdef cppclass shared_ptr[T]:
        shared_ptr()
        void reset()
        T* get()
        void swap(shared_ptr&)
        long use_count()
        bool unique()
        # not c++98...
        #bool operator bool()
        bool operator!()
    cdef cppclass weak_ptr[T]:
        weak_ptr()
        bool expired()
        shared_ptr[T] lock()

cdef extern from "<pv/security.h>" namespace "epics::pvAccess" nogil:
    cdef cppclass PeerInfo:
        string peer
        string transport
        string authority
        string realm
        string account
        set[string] roles
        unsigned transportVersion
        bool local
        bool identified

cdef extern from "<pv/pvAccess.h>" namespace "epics::pvAccess" nogil:
    cdef cppclass ChannelProvider:
        pass
    cdef cppclass ChannelRequester:
        shared_ptr[const PeerInfo] getPeerInfo() except+

cdef extern from "<pv/configuration.h>" namespace "epics::pvAccess" nogil:
    cdef cppclass Configuration:
        pass
    cdef cppclass ConfigurationBuilder:
        ConfigurationBuilder& add(const string& key, const string& value) except+
        ConfigurationBuilder& push_map() except+
        shared_ptr[Configuration] build() except+

cdef extern from "gwchannel.h" namespace "GWProvider" nogil:
    cdef struct ReportItem:
        string usname
        string dsname
        string transportPeer
        string transportAccount
        double transportTX
        double transportRX
        double operationTX
        double operationRX

cdef extern from "gwchannel.h" nogil:
    cdef cppclass GWChan:

        unsigned allow_put
        unsigned allow_rpc
        unsigned allow_uncached

        void disconnect()

    cdef struct GWStats:
        size_t ccacheSize
        size_t mcacheSize
        size_t banHostSize
        size_t banPVSize
        size_t banHostPVSize

    enum: GWSearchIgnore
    enum: GWSearchClaim
    enum: GWSearchBanHost
    enum: GWSearchBanPV
    enum: GWSearchBanHostPV

    cdef cppclass GWProvider(ChannelProvider):
        PyObject* handle
        @staticmethod
        shared_ptr[GWProvider] build(const string& name, const shared_ptr[Configuration]& conf) except+

        shared_ptr[GWProvider] shared_from_this() except+

        int test(const string& usname)
        shared_ptr[GWChan] connect(const string &dsname, const string &usname, const shared_ptr[ChannelRequester]& requester) except+

        void sweep() except+
        void disconnect(const string& usname) except+
        void clearBan() except+
        void cachePeek(set[string]& names) except+
        void stats(GWStats& stats)
        void report(vector[ReportItem]& us, vector[ReportItem]& ds, double& period) except+

        @staticmethod
        void prepare() except+

GWProvider.prepare()

cdef class InfoBase(object):
    cdef shared_ptr[const PeerInfo] info

    @property
    def peer(self):
        if <bool>self.info:
            return self.info.get().peer.decode('UTF-8')
        else:
            return u''

    @property
    def identified(self):
        if <bool>self.info:
            return self.info.get().identified
        else:
            return False

    @property
    def account(self):
        if <bool>self.info:
            return self.info.get().account.decode('UTF-8')
        else:
            return u''

    @property
    def roles(self):
        if <bool>self.info:
            return [role.decode('UTF-8') for role in self.info.get().roles]
        else:
            return []

cdef class CreateOp(InfoBase):
    cdef readonly bytes name
    cdef weak_ptr[ChannelRequester] requester
    cdef weak_ptr[GWProvider] provider
    cdef object __weakref__

    def create(self, bytes name=None):
        cdef shared_ptr[GWChan] gwchan
        cdef Channel chan = Channel()
        cdef string dsname = self.name
        cdef string usname = name or self.name
        cdef shared_ptr[ChannelRequester] requester = self.requester.lock()
        cdef shared_ptr[GWProvider] provider = self.provider.lock()

        if <bool>requester and <bool>provider:
            with nogil:
                gwchan = provider.get().connect(dsname, usname, requester)

            if not gwchan:
                raise RuntimeError("GW Provider will not create %s -> %s"%(usname, dsname))

            chan.channel = <weak_ptr[GWChan]>gwchan
            chan.info = self.info
            return chan
        else:
            raise RuntimeError("Dead CreateOp")


cdef class Channel(InfoBase):
    cdef readonly bytes name
    cdef weak_ptr[GWChan] channel
    cdef object __weakref__

    @property
    def expired(self):
        return self.channel.expired()

    def access(self, put=None, rpc=None, uncached=None):
        cdef shared_ptr[GWChan] ch = self.channel.lock()
        if not ch:
            return
        if put is not None:
            ch.get().allow_put = put==True
        if rpc is not None:
            ch.get().allow_rpc = rpc==True
        if uncached is not None:
            ch.get().allow_uncached = uncached==True

    def close(self):
        cdef shared_ptr[GWChan] ch = self.channel.lock()
        if <bool>ch:
            ch.get().disconnect()

@cython.no_gc_clear
cdef class Provider:
    cdef shared_ptr[GWProvider] provider
    cdef object __weakref__
    cdef object dummy # ensure that this type participates in GC

    cdef readonly int Claim
    cdef readonly int Ignore
    cdef readonly int BanHost
    cdef readonly int BanPV
    cdef readonly int BanHostPV

    def __cinit__(self):
        self.Claim = GWSearchClaim
        self.Ignore = GWSearchIgnore
        self.BanHost = GWSearchBanHost
        self.BanPV = GWSearchBanPV
        self.BanHostPV = GWSearchBanHostPV

    def testChannel(self, bytes usname):
        cdef string n = usname
        cdef int ret
        with nogil:
            ret = self.provider.get().test(n)
        return ret

    def sweep(self):
        with nogil:
            self.provider.get().sweep()

    def disconnect(self, bytes usname):
        cdef string n = usname
        with nogil:
            self.provider.get().disconnect(n)

    def clearBan(self):
        with nogil:
            self.provider.get().clearBan()

    def cachePeek(self):
        cdef set[string] ret

        self.provider.get().cachePeek(ret)
        return ret

    def stats(self):
        cdef GWStats stats
        self.provider.get().stats(stats)

        # cf. statsType in gw.py
        return {
            'ccacheSize.value':stats.ccacheSize,
            'mcacheSize.value':stats.mcacheSize,
            'banHostSize.value':stats.banHostSize,
            'banPVSize.value':stats.banPVSize,
            'banHostPVSize.value':stats.banHostPVSize,
        }

    def report(self):
        cdef vector[ReportItem] us
        cdef vector[ReportItem] ds
        cdef double period = 0.0 # initial value not used, but quiets warning
        cdef ReportItem item

        with nogil:
            self.provider.get().report(us, ds, period)

        rus = []
        for item in us:
            # order in tuple must match column order
            rus.append((
                item.usname.decode('UTF-8'),
                item.operationTX,
                item.operationRX,
                item.transportPeer.decode('UTF-8'),
                item.transportTX,
                item.transportRX,
            ))

        rds = []
        for item in ds:
            # order in tuple must match column order
            rds.append((
                item.usname.decode('UTF-8'),
                item.dsname.decode('UTF-8'),
                item.operationTX,
                item.operationRX,
                item.transportAccount.decode('UTF-8'),
                item.transportPeer.decode('UTF-8'),
                item.transportTX,
                item.transportRX,
            ))

        return rus, rds, period

    def use_count(self):
        return self.provider.use_count()

    def __dealloc__(self):
        with nogil:
            self.provider.reset()

# Allow GC to find handler stored in GWProvider
#   https://github.com/cython/cython/issues/2737
cdef traverseproc Provider_base_traverse

cdef int holder_traverse(PyObject* raw, visitproc visit, void* arg) except -1:
    cdef int ret = 0
    cdef Provider self = <Provider>raw

    if self.provider.get().handle:
        visit(self.provider.get().handle, arg)
    return Provider_base_traverse(raw, visit, arg)

Provider_base_traverse = (<PyTypeObject*>Provider).tp_traverse
(<PyTypeObject*>Provider).tp_traverse = <traverseproc>holder_traverse

def installGW(unicode name, dict config, object handler):
    cdef string cname = name.encode('utf-8')
    cdef ConfigurationBuilder builder
    cdef shared_ptr[GWProvider] provider
    cdef Provider ret = Provider()

    for K, V in config.items():
        builder.add(K.encode('utf-8'), V.encode('utf-8'))

    with nogil:
        provider = GWProvider.build(cname, builder.push_map().build())

    Py_INCREF(handler)
    Py_XDECREF(provider.get().handle)
    provider.get().handle = <PyObject*>handler

    ret.provider = provider
    return ret

# called from gwchannel.cpp
cdef public:
    void GWProvider_cleanup(GWProvider* provider) with gil:
        Py_XDECREF(provider.handle)

    int GWProvider_testChannel(GWProvider* provider, const char* name, const char* peer) with gil:
        if not provider.handle:
            return GWSearchBanHost
        handle = <object>provider.handle
        try:
            return handle.testChannel(name, peer.decode('UTF-8'))
        except:
            import traceback
            traceback.print_exc()
            return GWSearchBanHost

    shared_ptr[GWChan] GWProvider_makeChannel(GWProvider* provider, const string& name, const shared_ptr[ChannelRequester]& requester) with gil:
        cdef shared_ptr[GWChan] ret
        cdef CreateOp op
        cdef Channel result

        if provider.handle:
            op = CreateOp()
            handle = <object>provider.handle
            op.name = name.c_str()
            op.requester = <weak_ptr[ChannelRequester]>requester
            op.provider = <weak_ptr[GWProvider]>provider.shared_from_this()
            op.info = requester.get().getPeerInfo()

            try:
                chan = handle.makeChannel(op)
            except:
                import traceback
                traceback.print_exc()
                chan = None

            if chan is not None:
                ret = (<Channel?>chan).channel.lock()

        return ret
