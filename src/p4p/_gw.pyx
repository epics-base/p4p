# distutils: language = c++
#cython: language_level=2

cimport cython

from libc.stddef cimport size_t
from libc.string cimport strchr
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list as listxx
from libcpp.set cimport set as setxx

from cpython.object cimport PyObject, PyTypeObject, traverseproc, visitproc
from cpython.ref cimport Py_INCREF, Py_XDECREF

cdef extern from "<epicsAtomic.h>":
    cdef void atomic_set "::epics::atomic::set" (int& var, int val)

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
        setxx[string] roles
        unsigned transportVersion
        bool local
        bool identified

cdef extern from "<pv/pvAccess.h>" namespace "epics::pvAccess" nogil:
    cdef cppclass ChannelProvider:
        pass
    cdef cppclass ChannelRequester:
        shared_ptr[const PeerInfo] getPeerInfo() except+
    cdef cppclass ChannelProviderFactory:
        pass
    cdef cppclass ChannelProviderRegistry:
        @staticmethod
        shared_ptr[ChannelProviderRegistry] clients() except+
        shared_ptr[ChannelProviderFactory] remove(string& name) except+

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
    void GWInstallClientAliased(shared_ptr[ChannelProvider]& provider, string& installAs) except+

    cdef cppclass GWChan:

        int allow_put
        int allow_rpc
        int allow_uncached
        int audit
        int get_holdoff

        void disconnect()

    cdef struct GWStats:
        size_t ccacheSize
        size_t mcacheSize
        size_t gcacheSize
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
        shared_ptr[ChannelProvider] buildClient(const string& name, const shared_ptr[Configuration]& conf) except+
        @staticmethod
        shared_ptr[GWProvider] build(const string& name, const shared_ptr[ChannelProvider]& provider) except+

        shared_ptr[GWProvider] shared_from_this() except+

        int test(const string& usname)
        shared_ptr[GWChan] connect(const string &dsname, const string &usname, const shared_ptr[ChannelRequester]& requester) except+

        void sweep() except+
        void disconnect(const string& usname) except+
        void forceBan(const string& host, const string& usname) except+
        void clearBan() except+
        void cachePeek(setxx[string]& names) except+
        void stats(GWStats& stats)
        void report(vector[ReportItem]& us, vector[ReportItem]& ds, double& period) except+

        @staticmethod
        void prepare() except+

GWProvider.prepare()

cdef class ClientInstaller(object):
    cdef string name
    cdef weak_ptr[ChannelProvider] provider

    def __enter__(self):
        cdef shared_ptr[ChannelProvider] prov = self.provider.lock()
        with nogil:
            if <bool>prov:
                GWInstallClientAliased(prov, self.name)
            prov.reset()

    def __exit__(self,A,B,C):
        with nogil:
            ChannelProviderRegistry.clients().get().remove(self.name)

cdef class Client(object):
    """Client(provider, config)
    GW Client.  Wraps a C++ class ChannelProvider.  Pass to `Provider` ctor.

    ``config`` dict takes the same keys as `p4p.client.thread.Context` with the "pva" provider.

    :param unicode provider: Client provider name from global register
    :param dict config: Configuration for new client instance.
    """
    cdef shared_ptr[ChannelProvider] provider

    def __init__(self, unicode provider, dict config):
        cdef ConfigurationBuilder builder
        cdef string name = provider.encode('utf-8')

        for K, V in config.items():
            builder.add(K.encode('utf-8'), V.encode('utf-8'))

        with nogil:
            self.provider = GWProvider.buildClient(name, builder.push_map().build())

    def __dealloc__(self):
        with nogil:
            self.provider.reset()

    def installAs(self, unicode name):
        """Install in process-wide client registry under an alias (eg. not "pva")
        """
        cdef ClientInstaller inst = ClientInstaller()
        inst.name = name.encode('utf-8')
        inst.provider = <weak_ptr[ChannelProvider]>self.provider
        return inst

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
        cdef list ret = []
        if <bool>self.info:
            for role in self.info.get().roles:
                ret.append(role.decode('UTF-8'))
        return ret

    def __dealloc__(self):
        with nogil:
            self.info.reset()

cdef class CreateOp(InfoBase):
    """Handle for in-progress Channel creation request
    """
    cdef readonly bytes name
    cdef weak_ptr[ChannelRequester] requester
    cdef weak_ptr[GWProvider] provider
    cdef object __weakref__

    def create(self, bytes name=None):
        """Create a Channel with a given upstream (server-side) name

        :param bytes name: Upstream name to use.  This is what the GW Client will search for.
        :returns: A `Channel`
        """
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
    """Wraps C++ class GWChan

    cf. `CreateOp.create()`
    """
    cdef readonly bytes name
    cdef weak_ptr[GWChan] channel
    cdef object __weakref__

    @property
    def expired(self):
        """Has this Channel become unused/disconnected?
        """
        return self.channel.expired()

    def access(self, put=None, rpc=None, uncached=None, audit=None, holdoff=None):
        """Configure access control permissions, and other restrictions, on this channel.

        :param put: None to leave unchanged.  bool to permit/deny PUT operations
        :param rpc: None to leave unchanged.  bool to permit/deny RPC operations
        :param uncached: None to leave unchanged.  bool to permit/deny cache bypass for GET/MONITOR
        :param audit: None to leave unchanged.  bool to enable/disable PUT logging
        :param holdoff: None to leave unchanged.  float value to set GET holdoff period
        """
        cdef shared_ptr[GWChan] ch = self.channel.lock()
        if not ch:
            return
        if put is not None:
            atomic_set(ch.get().allow_put, put==True)
        if rpc is not None:
            atomic_set(ch.get().allow_rpc, rpc==True)
        if uncached is not None:
            atomic_set(ch.get().allow_uncached, uncached==True)
        if audit:
            atomic_set(ch.get().audit, audit==True)
        if holdoff:
            atomic_set(ch.get().get_holdoff, holdoff*1000)

    def close(self):
        """Force disconnect this Channel
        """
        cdef shared_ptr[GWChan] ch = self.channel.lock()
        if <bool>ch:
            ch.get().disconnect()

@cython.no_gc_clear
cdef class Provider:
    """Provider(name, client, handler)
    GW Server endpoint.  wrapper for C++ class GWProvider

    :param unicode name: Unique name of this provider
    :param Client client: Associated client to which requests are forwarded
    :param handler: Callbacks
    """
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

    def __init__(self, unicode name, Client client, object handler):
        cdef string cname = name.encode('utf-8')
        cdef shared_ptr[ChannelProvider] cprov

        cprov = client.provider
        with nogil:
            self.provider = GWProvider.build(cname, cprov)

        Py_INCREF(handler)
        Py_XDECREF(self.provider.get().handle)
        self.provider.get().handle = <PyObject*>handler

    def __dealloc__(self):
        with nogil:
            self.provider.reset()

    def testChannel(self, bytes usname):
        """testChannel(usname)
        Add the upstream name to the channel cache and begin trying to connect.
        Returns Claim if the channel is connected, and Ignore if it is not.

        :param bytes usname: Upstream (Server side) PV name
        :returns: Claim or Ignore
        """
        cdef string n = usname
        cdef int ret
        with nogil:
            ret = self.provider.get().test(n)
        return ret

    def sweep(self):
        """Call periodically to remove unused `Channel` from channel cache.
        """
        with nogil:
            self.provider.get().sweep()

    def disconnect(self, bytes usname):
        """Force disconnection of all channels connected to the named PV

        :param bytes usname: Upstream (Server side) PV name
        """
        cdef string n = usname
        with nogil:
            self.provider.get().disconnect(n)

    def forceBan(self, bytes host = None, bytes usname = None):
        """Preemptively Add an entry to the negative result cache.
        Either host or usname must be not None

        :param bytes host: None or a host name
        :param bytes usname: None or a upstream (Server side) PV name
        """
        cdef string h
        cdef string us
        if host:
            h = host
        if usname:
            us = usname
        with nogil:
            self.provider.get().forceBan(h, us)

    def clearBan(self):
        """Clear the negative results cache
        """
        with nogil:
            self.provider.get().clearBan()

    def cachePeek(self):
        """Returns PV names in channel cache

        :returns: a set of strings
        """
        cdef setxx[string] pvs
        cdef set ret

        self.provider.get().cachePeek(pvs)
        ret = set()
        for name in pvs:
            ret.add(name)
        return ret

    def stats(self):
        """Return statistics of various internal caches

        :rtype: dict
        """
        cdef GWStats stats
        self.provider.get().stats(stats)

        # cf. statsType in gw.py
        return {
            'ccacheSize.value':stats.ccacheSize,
            'mcacheSize.value':stats.mcacheSize,
            'gcacheSize.value':stats.gcacheSize,
            'banHostSize.value':stats.banHostSize,
            'banPVSize.value':stats.banPVSize,
            'banHostPVSize.value':stats.banHostPVSize,
        }

    def report(self):
        """Run bandwidth usage report

        :returns: A tuple of upstream (server side), downstream (client side), and time since last report() call.
        :rtype: ([(usname, opTx, opRx, peer, trTx, trRx)], [(usname, dsname, opTx, opRx, account, peer, trTx, trRx)], float)
        """
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

    void GWProvider_audit(GWProvider* provider, listxx[string]& audits) with gil:
        if provider.handle:
            handle = <object>provider.handle

            for audit in audits:
                try:
                    handle.audit(audit)
                except:
                    import traceback
                    traceback.print_exc()
                    break
