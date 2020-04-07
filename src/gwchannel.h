#ifndef GWCHANNEL_H
#define GWCHANNEL_H

#include <sstream>
#include <list>
#include <set>

#include <Python.h>

#include <epicsMutex.h>
#include <epicsGuard.h>
#include <epicsTime.h>
#include <epicsEvent.h>
#include <epicsAtomic.h>

#include <pv/timer.h>
#include <pv/thread.h>
#include <pv/valueBuilder.h>
#include <pv/pvAccess.h>
#include <pv/security.h>
#include <pv/configuration.h>
#include <pv/reftrack.h>

#include <pv/valueBuilder.h>
#include <pv/pvAccess.h>
#include <pv/security.h>
#include <pv/configuration.h>
#include <pv/reftrack.h>

// handle -fvisibility=default
// effects generated _gw.cpp
#if __GNUC__ >= 4
#  undef PyMODINIT_FUNC
#  if PY_MAJOR_VERSION < 3
#    define PyMODINIT_FUNC extern "C" __attribute__ ((visibility("default"))) void
#  else
#    define PyMODINIT_FUNC extern "C" __attribute__ ((visibility("default"))) PyObject*
#  endif
#endif

// enable extremely verbose low level debugging prints.
#if 0
#define TRACING
std::ostream& show_time(std::ostream&);
#define TRACE(ARG) do{ show_time(std::cerr<<"TRACE ")<<__FUNCTION__<<" "<<ARG<<"\n";} while(0)
#else
#define TRACE(ARG) do{ } while(0)
#endif

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

void GWInstallClientAliased(const pva::ChannelProvider::shared_pointer& provider,
                            const std::string& installAs);

struct GWProvider;

struct GWChan : public pva::Channel,
                public std::tr1::enable_shared_from_this<GWChan>
{
    POINTER_DEFINITIONS(GWChan);

    static size_t num_instances;

    struct Requester : public requester_type {
        POINTER_DEFINITIONS(Requester);
        static size_t num_instances;

        mutable epicsMutex mutex;

        // our corresponding (Client) channel
        pva::Channel::shared_pointer us_channel;
        // use map as one US channel could have many DS channels
        typedef std::map<GWChan*, GWChan::weak_pointer> weak_t;
        typedef std::vector<GWChan::shared_pointer> strong_t;
        weak_t ds_requesters;

        bool poked;

        Requester();
        virtual ~Requester();

        void latch(strong_t& chans);

        virtual std::string getRequesterName() OVERRIDE FINAL;
        virtual void message(std::string const & message,pva::MessageType messageType) OVERRIDE FINAL;
        virtual void channelCreated(const pvd::Status& status, pva::Channel::shared_pointer const & channel) OVERRIDE FINAL;
        virtual void channelStateChange(pva::Channel::shared_pointer const & channel, pva::Channel::ConnectionState connectionState) OVERRIDE FINAL;

        EPICS_NOT_COPYABLE(Requester)
    };

    // our name.  may differ from base_channel->getChannelName()
    const std::string name;
    // provide through which we were created
    const std::tr1::weak_ptr<GWProvider> provider;

    // const after GWProvider::createChannel()
    pva::Channel::shared_pointer us_channel;
    const pva::ChannelRequester::weak_pointer ds_requester;
    // requester given to base_channel
    std::tr1::shared_ptr<Requester> us_requester;

    // Use atomic access.
    // binary flags
    int allow_put,
        allow_rpc,
        allow_uncached,
        audit;
    // time in msec
    int get_holdoff;

    GWChan(const std::tr1::shared_ptr<GWProvider>& provider,
           const std::string& name,
           const pva::ChannelRequester::weak_pointer& requester);
    virtual ~GWChan();

    virtual void destroy() OVERRIDE FINAL;

    virtual std::tr1::shared_ptr<pva::ChannelProvider> getProvider() OVERRIDE FINAL;
    virtual std::string getRemoteAddress() OVERRIDE FINAL;
    virtual std::string getChannelName() OVERRIDE FINAL;
    virtual std::tr1::shared_ptr<pva::ChannelRequester> getChannelRequester() OVERRIDE FINAL;

    // force upstream disconnect.
    void disconnect();

    virtual void getField(pva::GetFieldRequester::shared_pointer const & requester,std::string const & subField) OVERRIDE FINAL;
    virtual pva::ChannelGet::shared_pointer createChannelGet(pva::ChannelGetRequester::shared_pointer const & requester,
                                                             pvd::PVStructure::shared_pointer const & pvRequest) OVERRIDE FINAL;
    virtual pva::ChannelPut::shared_pointer createChannelPut(pva::ChannelPutRequester::shared_pointer const & requester,
                                                             pvd::PVStructure::shared_pointer const & pvRequest) OVERRIDE FINAL;
    virtual pva::ChannelRPC::shared_pointer createChannelRPC(pva::ChannelRPCRequester::shared_pointer const & requester,
                                                             pvd::PVStructure::shared_pointer const & pvRequest) OVERRIDE FINAL;
    virtual pva::Monitor::shared_pointer createMonitor(pva::MonitorRequester::shared_pointer const & requester,
                                                       pvd::PVStructure::shared_pointer const & pvRequest) OVERRIDE FINAL;

    EPICS_NOT_COPYABLE(GWChan)
};

struct GWMon : public pva::MonitorFIFO
{
    POINTER_DEFINITIONS(GWMon);

    static size_t num_instances;

    struct Requester : public requester_type {
        POINTER_DEFINITIONS(Requester);
        static size_t num_instances;

        const std::string name;

        mutable epicsMutex mutex;

        pva::Monitor::shared_pointer us_op;

        // use map as one US monitor could have many DS monitors
        typedef std::map<GWMon*, GWMon::weak_pointer> weak_t;
        typedef std::vector<GWMon::shared_pointer> strong_t;
        weak_t ds_ops;

        pvd::PVStructure::shared_pointer complete;
        pvd::BitSet valid;

        pva::NetStats::Stats prevStats;

        explicit Requester(const std::string& usname);
        virtual ~Requester();

        void latch(strong_t& mons);

        virtual std::string getRequesterName() OVERRIDE FINAL;
        virtual void channelDisconnect(bool destroy) OVERRIDE FINAL;
        virtual void monitorConnect(pvd::Status const & status,
                                    pva::MonitorPtr const & monitor, pvd::StructureConstPtr const & structure) OVERRIDE FINAL;
        virtual void monitorEvent(pva::MonitorPtr const & monitor) OVERRIDE FINAL;
        virtual void unlisten(pva::MonitorPtr const & monitor) OVERRIDE FINAL;

        EPICS_NOT_COPYABLE(Requester)
    };

    const std::string name; // copy of corresponding GWChan::name

    // const after GWChan::createMonitor()
    GWMon::Requester::shared_pointer us_requester;
    GWChan::shared_pointer channel;

    // not guarded (not reentrant)
    pva::NetStats::Stats prevStats;

    GWMon(const std::string& name,
          const pva::MonitorRequester::shared_pointer& requester,
          const pvd::PVStructure::const_shared_pointer &pvRequest,
          const Source::shared_pointer& source = Source::shared_pointer(),
          Config *conf=0);
    virtual ~GWMon();

    EPICS_NOT_COPYABLE(GWMon)
};

struct ProxyGet : public pva::ChannelGet,
                  public std::tr1::enable_shared_from_this<ProxyGet>
{
    POINTER_DEFINITIONS(ProxyGet);
    typedef ProxyGet gw_operation_type;

    static size_t num_instances;

    struct Requester : public requester_type,
                       public pvd::TimerCallback,
                       public std::tr1::enable_shared_from_this<Requester>
    {
        static size_t num_instances;

        POINTER_DEFINITIONS(Requester);

        const std::tr1::shared_ptr<struct GWChan> channel;

        mutable epicsMutex mutex;

        pva::ChannelGet::shared_pointer us_op;

        // requesters from downstream (PVA server)
        // use map as one US get could be shared by many DS gets
        typedef std::map<gw_operation_type*, gw_operation_type::weak_pointer> weak_t;
        typedef std::vector<gw_operation_type::shared_pointer> strong_t;
        weak_t ds_ops;

        enum state_t {
            // waiting for upstream channelGetConnect()
            Disconnected,
            // waiting for downstream get()
            Idle,
            // waiting for upstream getDone()
            Executing,
            // waiting for holdoff timer to expire
            Holdoff,
            // waiting for holdoff timer to expire, after a downstream get()
            HoldoffQueued,
            // channelGetConnect() errors
            Dead,
        } state;

        // state==Disconnected implies !type
        pvd::Structure::const_shared_pointer type;

        explicit Requester(const std::tr1::shared_ptr<struct GWChan>& channel);
        virtual ~Requester();

        bool latch(strong_t& mons, bool reset=false, bool onlybusy=false);

        virtual std::string getRequesterName() OVERRIDE FINAL;
        virtual void message(std::string const & message,pva::MessageType messageType) OVERRIDE FINAL;
        virtual void channelDisconnect(bool destroy) OVERRIDE FINAL;

        virtual void channelGetConnect(const pvd::Status& status,
                                       pva::ChannelGet::shared_pointer const & channelGet,
                                       pvd::Structure::const_shared_pointer const & structure) OVERRIDE FINAL;

        virtual void getDone(const pvd::Status& status,
                             pva::ChannelGet::shared_pointer const & channelGet,
                             pvd::PVStructure::shared_pointer const & pvStructure,
                             pvd::BitSet::shared_pointer const & bitSet) OVERRIDE FINAL;

        virtual void callback() OVERRIDE FINAL;
        virtual void timerStopped() OVERRIDE FINAL;
    };
    // requester given to us_op
    const Requester::shared_pointer us_requester;

    const requester_type::weak_pointer ds_requester;

    const pvd::PVStructurePtr ds_pvRequest;

    pvd::PVRequestMapper mapper;

    // protected by us_requester->mutex

    // downstream requests get()
    bool executing;

    ProxyGet(const Requester::shared_pointer& us_requester,
             const requester_type::shared_pointer& ds_requester,
             const pvd::PVStructurePtr& ds_pvRequest);
    virtual ~ProxyGet();

    virtual void destroy() OVERRIDE FINAL;
    virtual std::tr1::shared_ptr<pva::Channel> getChannel() OVERRIDE FINAL;
    virtual void cancel() OVERRIDE FINAL;
    virtual void lastRequest() OVERRIDE FINAL;

    virtual void get() OVERRIDE FINAL;
};

// intercept execution to perform authorization check.
// no need to intercept *Requester
struct ProxyPut : public pva::ChannelPut,
                  public std::tr1::enable_shared_from_this<ProxyPut>
{
    POINTER_DEFINITIONS(ProxyPut);
    typedef ProxyPut gw_operation_type;

    static size_t num_instances;

    struct Requester : public requester_type {
        static size_t num_instances;

        POINTER_DEFINITIONS(Requester);

        // requester from downstream (PVA server)
        const requester_type::weak_pointer downstream;
        // operation given to downstream.
        // const after GWChan::createChannel*()
        gw_operation_type::weak_pointer operation;

        Requester(const requester_type::weak_pointer& downstream);
        virtual ~Requester();

        virtual std::string getRequesterName() OVERRIDE FINAL;
        virtual void message(std::string const & message,pva::MessageType messageType) OVERRIDE FINAL;
        virtual void channelDisconnect(bool destroy) OVERRIDE FINAL;
        virtual void channelPutConnect(const pvd::Status& status,
                                       ChannelPut::shared_pointer const & channelPut,
                                       pvd::Structure::const_shared_pointer const & structure) OVERRIDE FINAL;
        virtual void putDone(const pvd::Status& status,
                             ChannelPut::shared_pointer const & channelPut) OVERRIDE FINAL;
        virtual void getDone(const pvd::Status& status,
                             ChannelPut::shared_pointer const & channelPut,
                             pvd::PVStructure::shared_pointer const & pvStructure,
                             pvd::BitSet::shared_pointer const & bitSet) OVERRIDE FINAL;

        EPICS_NOT_COPYABLE(Requester)
    };

    const std::tr1::shared_ptr<struct GWChan> channel;
    // requester given to us_op
    const Requester::shared_pointer us_requester;

    mutable epicsMutex mutex;

    pva::ChannelPut::shared_pointer us_op;

    ProxyPut(const std::tr1::shared_ptr<struct GWChan>& channel,
             const requester_type::shared_pointer& requester);
    virtual ~ProxyPut();

    virtual void destroy() OVERRIDE FINAL { us_op->destroy(); }
    virtual std::tr1::shared_ptr<pva::Channel> getChannel() OVERRIDE FINAL { return channel; }
    virtual void cancel() OVERRIDE FINAL { us_op->cancel(); }
    virtual void lastRequest() OVERRIDE FINAL { us_op->lastRequest(); }

    virtual void put(pvd::PVStructure::shared_pointer const & pvPutStructure,
                     pvd::BitSet::shared_pointer const & putBitSet) OVERRIDE FINAL;
    virtual void get() OVERRIDE FINAL { us_op->get(); }

    EPICS_NOT_COPYABLE(ProxyPut)
};


// intercept execution to perform authorization check.
// no need to intercept *Requester
struct ProxyRPC : public pva::ChannelRPC,
                  public std::tr1::enable_shared_from_this<ProxyRPC>
{
    POINTER_DEFINITIONS(ProxyRPC);
    typedef ProxyRPC gw_operation_type;

    static size_t num_instances;

    struct Requester : public requester_type {
        static size_t num_instances;

        POINTER_DEFINITIONS(Requester);

        // requester from downstream (PVA server)
        const requester_type::weak_pointer downstream;
        // operation given to downstream.
        // const after GWChan::createChannel*()
        gw_operation_type::weak_pointer operation;

        Requester(const requester_type::weak_pointer& downstream);
        virtual ~Requester();

        virtual std::string getRequesterName() OVERRIDE FINAL;
        virtual void message(std::string const & message,pva::MessageType messageType) OVERRIDE FINAL;
        virtual void channelDisconnect(bool destroy) OVERRIDE FINAL;
        virtual void channelRPCConnect(const pvd::Status& status,
                                       ChannelRPC::shared_pointer const & rpc) OVERRIDE FINAL;
        virtual void requestDone(const pvd::Status& status,
                                 ChannelRPC::shared_pointer const & rpc,
                                 pvd::PVStructure::shared_pointer const & pvResponse) OVERRIDE FINAL;

        EPICS_NOT_COPYABLE(Requester)
    };

    const std::tr1::shared_ptr<struct GWChan> channel;
    // requester given to us_op
    const Requester::shared_pointer us_requester;

    mutable epicsMutex mutex;

    pva::ChannelRPC::shared_pointer us_op;

    ProxyRPC(const std::tr1::shared_ptr<struct GWChan>& channel,
             const requester_type::shared_pointer& requester);
    ~ProxyRPC();

    virtual void destroy() OVERRIDE FINAL { us_op->destroy(); }
    virtual std::tr1::shared_ptr<pva::Channel> getChannel() OVERRIDE FINAL { return channel; }
    virtual void cancel() OVERRIDE FINAL { us_op->cancel(); }
    virtual void lastRequest() OVERRIDE FINAL { us_op->lastRequest(); }

    virtual void request(pvd::PVStructure::shared_pointer const & pvArgument) OVERRIDE FINAL;

    EPICS_NOT_COPYABLE(ProxyRPC)
};

enum GWSearchResult {
    GWSearchIgnore,
    GWSearchClaim,
    GWSearchBanHost,
    GWSearchBanPV,
    GWSearchBanHostPV,
};

struct GWStats {
    size_t ccacheSize,
           mcacheSize,
           gcacheSize,
           banHostSize,
           banPVSize,
           banHostPVSize;
};

struct GWProvider : public pva::ChannelProvider,
                    public std::tr1::enable_shared_from_this<GWProvider>
{
    POINTER_DEFINITIONS(GWProvider);

    static size_t num_instances;

    const std::string name;
    const pva::ChannelProvider::shared_pointer client;

    const pvd::PVStringArray::const_svector empty;

    // const after build()
    pva::ChannelFind::shared_pointer dummyFind;

    mutable epicsMutex mutex;

    typedef std::set<std::string> ban_t;
    ban_t banHost,
          banPV;
    std::set<std::pair<std::string, std::string> > banHostPV;

    typedef std::map<std::string, std::tr1::shared_ptr<GWChan::Requester> > channels_t;
    channels_t channels;

    typedef std::map<std::string, std::tr1::weak_ptr<GWMon::Requester> > monitors_t;
    monitors_t monitors;

    typedef std::map<std::string, std::tr1::shared_ptr<ProxyGet::Requester> > gets_t;
    gets_t gets;

    epicsTime prevtime;

    typedef std::list<std::string> audit_log_t;
    audit_log_t audit_log;
    epicsEvent audit_wakeup, audit_holdoff;
    bool audit_run;

    pvd::Thread audit_runner;

    pvd::Timer timerQueue;

    // guarded by GIL
    PyObject* handle;

private:
    GWProvider(const std::string& name, const pva::ChannelProvider::shared_pointer& provider);
public:
    virtual ~GWProvider();
    static std::tr1::shared_ptr<GWProvider> build(const std::string& name, const pva::ChannelProvider::shared_pointer& provider);

    static pva::ChannelProvider::shared_pointer buildClient(const std::string& name, const pva::Configuration::shared_pointer& conf);

    GWSearchResult test(const std::string& usname);

    GWChan::shared_pointer connect(const std::string& dsname,
                                   const std::string& usname,
                                   const pva::ChannelRequester::shared_pointer& requester);

    virtual void destroy() OVERRIDE FINAL;
    virtual std::string getProviderName() OVERRIDE FINAL;
    virtual pva::ChannelFind::shared_pointer channelFind(std::string const & name,
                                                         pva::ChannelFindRequester::shared_pointer const & requester) OVERRIDE FINAL;
    virtual pva::ChannelFind::shared_pointer channelList(pva::ChannelListRequester::shared_pointer const & requester) OVERRIDE FINAL;
    virtual pva::Channel::shared_pointer createChannel(std::string const & name,
                                                       pva::ChannelRequester::shared_pointer const & requester,
                                                       short priority, std::string const & address) OVERRIDE FINAL;

    void sweep();
    void disconnect(const std::string& usname);
    void forceBan(const std::string& host, const std::string& usname);
    void clearBan();

    void cachePeek(std::set<std::string> &names) const;

    void stats(GWStats& stats) const;

    struct ReportItem {
        std::string usname,
                    dsname;
        std::string transportPeer,
                    transportAccount;
        double transportTX, transportRX,
               operationTX, operationRX;
    };
    typedef std::vector<ReportItem> report_t;

    void report(report_t& us, report_t& ds, double& period);

    static void prepare();

private:
    void runAudit();

    EPICS_NOT_COPYABLE(GWProvider)
};

#endif // GWCHANNEL_H
