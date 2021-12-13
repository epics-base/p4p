#ifndef PVXS_GW_H
#define PVXS_GW_H

#ifndef PVXS_ENABLE_EXPERT_API
#  define PVXS_ENABLE_EXPERT_API
#endif

#include <atomic>

#include "p4p.h"

#include <epicsThread.h>

#include <pvxs/source.h>
#include <pvxs/sharedpv.h>
#include <pvxs/client.h>
#include <pvxs/util.h>

namespace p4p {
using namespace pvxs;

struct GWChan;
struct GWSource;

enum GWSearchResult {
    GWSearchIgnore,
    GWSearchClaim,
    GWSearchBanHost,
    GWSearchBanPV,
    GWSearchBanHostPV,
};

struct GWSubscription {
    //const std::shared_ptr<GWUpstream> chan;

    // should only be lock()'d from server worker
    std::weak_ptr<client::Subscription> upstream;

    Value current;

    enum state_t {
        Connecting, // waiting for onInit()
        Running,
        Error,
    } state = Connecting;

    std::vector<std::shared_ptr<server::MonitorSetupOp>> setups;
    std::vector<std::shared_ptr<server::MonitorControlOp>> controls;
};

struct GWGet {
    std::weak_ptr<client::Operation> upstream;

    Value prototype;
    epicsTime lastget;
    Timer delay;
    std::string error;

    enum state_t {
        Connecting, // waiting for onInit()
        Idle,
        Exec,
        Error,
    } state = Connecting;

    bool firstget = true;

    std::vector<std::shared_ptr<server::ConnectOp>> setups;
    std::vector<std::shared_ptr<server::ExecOp>> ops;
};

struct GWUpstream {
    const std::string usname;
    client::Context upstream;
    GWSource& src;

    mutable epicsMutex dschans_lock;
    std::set<std::shared_ptr<server::ChannelControl>> dschans;

    // time in msec
    std::atomic<unsigned> get_holdoff{};

    epicsMutex lock;

    std::weak_ptr<GWSubscription> subscription;

    std::weak_ptr<GWGet> getop;

    bool gcmark = false;

    const std::shared_ptr<MPMCFIFO<std::function<void()>>> workQ;

    const std::shared_ptr<client::Connect> connector;

    explicit GWUpstream(const std::string& usname, GWSource &src);
    ~GWUpstream();
};

struct GWChanInfo : public server::ReportInfo {
    std::string usname;
    explicit GWChanInfo(const std::string& usname) :usname(usname) {}
};
// workaround Cython limitations wrt. dynamic_cast[]
typedef const GWChanInfo* GWChanInfoCP;

struct GWChan {
    const std::string dsname;
    const std::shared_ptr<GWUpstream> us;
    const std::shared_ptr<server::ChannelControl> dschannel;
    const std::shared_ptr<const GWChanInfo> reportInfo;

    // Use atomic access.
    // binary flags
    std::atomic<bool> allow_put{},
                      allow_rpc{},
                      allow_uncached{},
                      audit{};


    GWChan(const std::string& usname,
           const std::string& dsname,
           const std::shared_ptr<GWUpstream>& upstream,
           const std::shared_ptr<server::ChannelControl>& dschannel);
    ~GWChan();

    static
    void onRPC(const std::shared_ptr<GWChan>& self, std::unique_ptr<server::ExecOp>&& op, Value&& arg);
    static
    void onOp(const std::shared_ptr<GWChan>& self, std::unique_ptr<server::ConnectOp>&& sop);
    static
    void onSubscribe(const std::shared_ptr<GWChan>& self, std::unique_ptr<server::MonitorSetupOp>&& sop);
};

struct AuditEvent {
    epicsTime now;
    std::string usname;
    std::string dsname;
    Value val;
    std::shared_ptr<const server::ClientCredentials> cred;
};

struct GWSource : public server::Source,
                  public std::enable_shared_from_this<GWSource>,
                  private epicsThreadRunable
{
    client::Context upstream;

    mutable epicsMutex mutex;

    std::set<std::string> banHost, banPV;
    std::set<std::pair<std::string, std::string>> banHostPV;

    PyObject *handler = nullptr;

    // channel cache.  Indexed by upstream name
    std::map<std::string, std::shared_ptr<GWUpstream>> channels;

    std::list<AuditEvent> audits;

    decltype (GWUpstream::workQ) workQ;

    epicsThread workQworker;

    static
    std::shared_ptr<GWSource> build(const client::Context& ctxt) {
        return std::shared_ptr<GWSource>(new GWSource(ctxt));
    }
    GWSource(const client::Context& ctxt);
    virtual ~GWSource();

    // for server::Source
    virtual void onSearch(Search &op) override final;
    virtual void onCreate(std::unique_ptr<server::ChannelControl> &&op) override final;

    GWSearchResult test(const std::string& usname);

    std::shared_ptr<GWChan> connect(const std::string& dsname,
                                    const std::string& usname,
                                    std::unique_ptr<server::ChannelControl> *op);

    void sweep();
    void disconnect(const std::string& usname);
    void forceBan(const std::string& host, const std::string& usname);
    void clearBan();

    void cachePeek(std::set<std::string> &names) const;

    void auditPush(AuditEvent&& evt);

    virtual void run() override final;
};

} // namespace p4p

#endif // PVXS_GW_H
