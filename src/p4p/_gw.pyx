# distutils: language = c++
#cython: language_level=2

cimport cython

from cpython.object cimport PyObject, PyTypeObject, traverseproc, visitproc
from cpython.ref cimport Py_INCREF, Py_XDECREF, Py_CLEAR

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr, shared_ptr, weak_ptr
from libcpp.string cimport string
from libcpp.list cimport list as listxx
from libcpp.set cimport set as setxx
from libcpp.vector cimport vector

from .pvxs.client cimport Context, Report, ReportInfo
from .pvxs.server cimport ServerGUID
from .pvxs.source cimport Source, ChannelControl, OpBase, ClientCredentials
from . cimport _p4p

cdef extern from "pvxs_gw.h" namespace "p4p" nogil:
    enum: GWSearchIgnore
    enum: GWSearchClaim
    enum: GWSearchBanHost
    enum: GWSearchBanPV
    enum: GWSearchBanHostPV

    cdef cppclass GWChanInfo(ReportInfo):
        string usname

    ctypedef const GWChanInfo* GWChanInfoCP

    cdef cppclass GWUpstream:
        unsigned get_holdoff

    cdef cppclass GWChan:
        const shared_ptr[GWUpstream] us
        const shared_ptr[ChannelControl] dschannel
        bool allow_put
        bool allow_rpc
        bool allow_uncached
        bool audit

    cdef cppclass GWSource(Source):
        Context upstream
        PyObject* handler

        @staticmethod
        shared_ptr[GWSource] build(const Context&) except+

        int test(const string&) except+
        shared_ptr[GWChan] connect(const string &dsname, const string &usname, unique_ptr[ChannelControl]* op) except+

        void sweep() except+
        void disconnect(const string& usname) except+
        void forceBan(const string& host, const string& usname) except+
        void clearBan() except+
        void cachePeek(setxx[string]& names) except+

        shared_ptr[GWSource] shared_from_this() except+

    shared_ptr[Source] makeOdometer(const string& name) except+

def addOdometer(_p4p.Server serv, basestring pvname, int order):
    cdef string name = pvname.encode()
    with nogil:
        serv.serv.addSource(name, makeOdometer(name), order)

cdef class InfoBase(object):
    cdef shared_ptr[const ClientCredentials] info

    @property
    def peer(self):
        if <bool>self.info:
            return self.info.get().peer.decode('UTF-8')
        else:
            return u''

    @property
    def account(self):
        if <bool>self.info:
            return self.info.get().account.decode('UTF-8')
        else:
            return u''

    @property
    def roles(self):
        cdef setxx[string] raw
        cdef list ret = []
        if <bool>self.info:
            for role in self.info.get().roles():
                ret.append(role.decode('UTF-8'))
        return ret

cdef class CreateOp(InfoBase):
    """Handle for in-progress Channel creation request
    """
    cdef readonly bytes name
    cdef unique_ptr[ChannelControl]* op
    cdef weak_ptr[GWSource] provider
    cdef object __weakref__

    def create(self, bytes name):
        """Create a Channel with a given upstream (server-side) name

        :param bytes name: Upstream name to use.  This is what the GW Client will search for.
        :returns: A `Channel`
        """
        cdef shared_ptr[GWChan] gwchan
        cdef Channel chan = Channel.__new__(Channel)
        cdef string dsname = self.name
        cdef string usname = name

        provider = self.provider.lock()
        if <bool>self.op and <bool>provider:
            with nogil:
                gwchan = provider.get().connect(dsname, usname, self.op)

            if not gwchan:
                raise RuntimeError("GW Provider will not create %s -> %s"%(usname, dsname))

            chan.channel = gwchan
            chan.name = name
            chan.info = gwchan.get().dschannel.get().credentials()
            return chan
        else:
            raise RuntimeError("Dead CreateOp")
            

cdef class Channel(InfoBase):
    cdef readonly bytes name
    cdef shared_ptr[GWChan] channel
    cdef object __weakref__

    @property
    def expired(self):
        return self.channel.use_count()<=1

    def access(self, put=None, rpc=None, uncached=None, audit=None, holdoff=None):
        if put is not None:
            self.channel.get().allow_put = put==True
        if rpc is not None:
            self.channel.get().allow_rpc = rpc==True
        if uncached is not None:
            self.channel.get().allow_uncached = uncached==True
        if audit is not None:
            self.channel.get().audit = audit==True
        if holdoff is not None:
            self.channel.get().us.get().get_holdoff = holdoff*1000 # sec -> ms

@cython.no_gc_clear
cdef class Provider(_p4p.Source):
    cdef shared_ptr[GWSource] provider
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

    def __init__(self, unicode name, object client, object handler):
        cdef _p4p.ClientProvider prov = client._ctxt
        cdef string cname = name.encode('utf-8')
        self.name = cname

        if not prov:
            raise ValueError('Not a Context')
        with nogil:
            self.provider = GWSource.build(prov.ctxt)
            self.src = <shared_ptr[Source]>self.provider

        Py_INCREF(handler)
        self.provider.get().handler = <PyObject*>handler

    def __dealloc__(self):
        if <bool>self.provider:
            Py_CLEAR(self.provider.get().handler)
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

    def ignoreByGUID(self, list servers):
        cdef _p4p.Server serv
        cdef vector[ServerGUID] guids

        for ent in servers:
            serv = <_p4p.Server?>ent
            guids.push_back(serv.serv.config().guid)

        self.provider.get().upstream.ignoreServerGUIDs(guids)

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
        # TODO
        return {
            'ccacheSize.value':0,
            'mcacheSize.value':0,
            'gcacheSize.value':0,
            'banHostSize.value':0,
            'banPVSize.value':0,
            'banHostPVSize.value':0,
        }

    def report(self, float norm=1.0):
        """Run Client/Upstream bandwidth usage report

        :returns: List of tuple
        :rtype: [(usname, opTx, opRx, peer, trTx, trRx)]
        """
        cdef Report report
        cdef double cnorm = norm
        with nogil:
            report = self.provider.get().upstream.report()

        usinfo = []
        for conn in report.connections:
            peer = conn.peer.decode()
            for chan in conn.channels:
                usinfo.append((chan.name.decode(), chan.tx/cnorm, chan.rx/cnorm, peer, conn.tx/cnorm, conn.rx/cnorm))

        return usinfo

    def use_count(self):
        return self.provider.use_count()

def Server_report(_p4p.Server serv, float norm=1.0):
    """Return Server/Downstream bandwidth usage report

    :returns: List of tuple
    :rtype: [(usname, dsname, opTx, opRx, account, peer, trTx, trRx)]
    """
    cdef GWChanInfoCP info
    cdef Report report
    cdef double cnorm = norm
    with nogil:
        report = serv.serv.report()

    dsinfo = []
    for conn in report.connections:
        peer = conn.peer.decode()
        if <bool>conn.credentials:
            account = conn.credentials.get().account.decode()
        else:
            account = ''

        for chan in conn.channels:
            info = dynamic_cast[GWChanInfoCP](chan.info.get())

            if info:
                usname = info.usname.decode()
            else:
                usname = ''

            dsinfo.append((usname, chan.name.decode(), chan.tx/cnorm, chan.rx/cnorm, account, peer, conn.tx/cnorm, conn.rx/cnorm))

    return dsinfo

# Allow GC to find handler stored in GWProvider
#   https://github.com/cython/cython/issues/2737
cdef traverseproc Provider_base_traverse

cdef int holder_traverse(PyObject* raw, visitproc visit, void* arg) except -1:
    cdef int ret = 0
    cdef Provider self = <Provider>raw

    if self.provider.get().handler:
        visit(self.provider.get().handler, arg)
    return Provider_base_traverse(raw, visit, arg)

Provider_base_traverse = (<PyTypeObject*>Provider).tp_traverse
(<PyTypeObject*>Provider).tp_traverse = <traverseproc>holder_traverse

cdef public:
    int GWProvider_testChannel(object handler, const char* channel, const char* peer) with gil:
        if not handler:
            return GWSearchBanHost
        try:
            return handler.testChannel(channel, peer.decode())
        except:
            return GWSearchBanHost

    shared_ptr[GWChan] GWProvider_makeChannel(GWSource* src, unique_ptr[ChannelControl]* op) with gil:
        cdef shared_ptr[GWChan] ret
        cdef CreateOp create
        if not src.handler:
            return ret

        handler = <object>src.handler

        create = CreateOp.__new__(CreateOp)
        create.name = op.get().name().c_str()
        create.op = op
        create.info = op.get().credentials()
        create.provider = <weak_ptr[GWSource]>src.shared_from_this()

        chan = handler.makeChannel(create)

        create.op = NULL # makeChannel() must decide immediately

        if chan is not None:
            ret = (<Channel?>chan).channel

        return ret

    void GWProvider_audit(GWSource* src, listxx[string]& cmsgs) with gil:
        if src.handler:
            handler = <object>src.handler
            msgs = [msg.decode() for msg in cmsgs]
            try:
                handler.audit(msgs)
            except:
                pass # expect handler to catch and log
