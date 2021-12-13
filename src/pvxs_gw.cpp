
#ifndef PVXS_ENABLE_EXPERT_API
#  define PVXS_ENABLE_EXPERT_API
#endif

#include "p4p.h"

#include <pvxs/source.h>
#include <pvxs/sharedpv.h>
#include <pvxs/client.h>
#include <pvxs/log.h>

#include "pvxs_gw.h"
#include "_gw.h"

DEFINE_LOGGER(_log, "p4p.gw");
DEFINE_LOGGER(_logget, "p4p.gw.get");
DEFINE_LOGGER(_logmon, "p4p.gw.sub");

namespace p4p {

GWSource::GWSource(const client::Context& ctxt)
    :upstream(ctxt)
    ,workQ(std::make_shared<decltype(workQ)::element_type>())
    ,workQworker(*this, "GWQ",
                 epicsThreadGetStackSize(epicsThreadStackBig),
                 epicsThreadPriorityMedium)
{
    workQworker.start();
}

GWSource::~GWSource() {
    workQ->push(nullptr);
    workQworker.exitWait();
}

void GWSource::onSearch(Search &op)
{
    // on server worker

    Guard G(mutex);

    decltype (banHostPV)::value_type pair;
    pair.first = op.source();

    if(banHost.find(pair.first)!=banHost.end()) {
        log_debug_printf(_log, "%p ignore banned host '%s'\n", this, pair.first.c_str());
        return;
    }

    for(auto& chan : op) {
        pair.second = chan.name();

        if(banPV.find(pair.second)!=banPV.end()) {
            log_debug_printf(_log, "%p ignore banned PV '%s'\n", this, pair.second.c_str());
            continue;
        } else if(banHost.find(pair.first)!=banHost.end()) {
            log_debug_printf(_log, "%p ignore banned Host '%s'\n", this, pair.first.c_str());
            continue;
        } else if(banHostPV.find(pair)!=banHostPV.end()) {
            log_debug_printf(_log, "%p ignore banned Host+PV '%s':'%s'\n", this, pair.first.c_str(), pair.second.c_str());
            continue;
        }

        GWSearchResult result = GWSearchIgnore;
        {
            // GWProvider_testChannel() will also lock our mutex, but must unlock first
            // to maintain lock order ordering wrt. GIL.
            UnGuard U(G);
            PyLock L;

            result = (GWSearchResult)GWProvider_testChannel(handler, chan.name(), op.source());
        }
        log_debug_printf(_log, "%p testChannel '%s':'%s' -> %d\n", this, pair.first.c_str(), pair.second.c_str(), result);

        switch(result) {
        case GWSearchClaim:
            chan.claim();
            break;
        case GWSearchBanHost:
            banHost.insert(pair.first);
            break;
        case GWSearchBanPV:
            banPV.insert(pair.second);
            break;
        case GWSearchBanHostPV:
            banHostPV.insert(pair);
            break;
        case GWSearchIgnore:
            break;
        }
    }
}

GWUpstream::GWUpstream(const std::string& usname, GWSource &src)
    :usname(usname)
    ,upstream(src.upstream)
    ,src(src)
    ,workQ(src.workQ)
    ,connector(upstream.connect(usname)
               .onConnect([this](){
                    log_debug_printf(_log, "%p upstream connect '%s'\n", &this->src, this->usname.c_str());
                })
               .onDisconnect([this]()
                {
                    // on client worker
                    log_debug_printf(_log, "%p upstream disconnect '%s'\n", &this->src, this->usname.c_str());
                    Guard G(dschans_lock);
                    for(auto& chan : dschans) {
                        chan->close();
                    }
                })
               .exec())
{}

GWUpstream::~GWUpstream()
{
    log_debug_printf(_log, "upstream shutdown %s\n", usname.c_str());
    Guard G(dschans_lock);
    for(auto& chan : dschans) {
        chan->close();
    }
}

GWChan::GWChan(const std::string& usname,
               const std::string& dsname,
               const std::shared_ptr<GWUpstream>& upstream,
               const std::shared_ptr<server::ChannelControl> &dschannel)
    :dsname(dsname)
    ,us(upstream)
    ,dschannel(dschannel)
    ,reportInfo(std::make_shared<const GWChanInfo>(usname))
{
    log_debug_printf(_log, "GWChan create %s\n", dsname.c_str());
    Guard G(us->dschans_lock);
    us->dschans.insert(dschannel);
}

GWChan::~GWChan()
{
    log_debug_printf(_log, "GWChan destroy %s\n", dsname.c_str());
    Guard G(us->dschans_lock);
    us->dschans.erase(dschannel);
}

void GWChan::onRPC(const std::shared_ptr<GWChan>& pv, std::unique_ptr<server::ExecOp> &&op, Value &&arg)
{
    // on server worker

    std::shared_ptr<server::ExecOp> sop(std::move(op));

    bool permit = pv->allow_rpc;

    log_debug_printf(_log, "'%s' RPC %s\n", sop->name().c_str(), permit ? "begin" : "DENY");

    if(!permit) {
        op->error("RPC permission denied by gateway");
        return;
    }

    auto cliop = pv->us->upstream.rpc(pv->us->usname, arg)
            .syncCancel(false)
            .result([sop](client::Result&& result)
    {
        // on client worker

        log_debug_printf(_log, "'%s' RPC complete\n", sop->name().c_str());

        // syncs client worker with server worker
        try {
            sop->reply(result());
        }catch(client::RemoteError& e) {
            sop->error(e.what());
        }catch(std::exception& e) {
            log_err_printf(_log, "RPC error: %s\n", e.what());
            sop->error(std::string("Error: ")+e.what());
        }
    })
            .exec();

    // just need to keep client op alive
    sop->onCancel([cliop]() {});
}

static
void onInfo(const std::shared_ptr<GWChan>& pv, const std::shared_ptr<server::ConnectOp>& ctrl)
{
    // on server worker
    // 1. downstream creating INFO operation

    log_debug_printf(_log, "'%s' INFO\n", ctrl->name().c_str()); // ============ INFO

    auto cliop = pv->us->upstream.info(pv->us->usname)
            .syncCancel(false)
            .result([ctrl](client::Result&& result)
    {
        // on client worker

        log_debug_printf(_log, "'%s' GET INFO done\n", ctrl->name().c_str());

        try{
            ctrl->connect(result());
        }catch(std::exception& e){
            ctrl->error(e.what());
            return;
        }
    })
            .exec();

    // just need to keep client op alive
    ctrl->onClose([cliop](const std::string&) {
        log_debug_printf(_log, "info close '%s'\n", cliop->name().c_str());
    });

}

static
void onGetCached(const std::shared_ptr<GWChan>& pv, const std::shared_ptr<server::ConnectOp>& ctrl)
{
    // on server worker
    // 1. downstream creating GET operation, maybe with holdoff

    auto us(pv->us);

    Guard G(us->lock);
    auto get(us->getop.lock());
    auto cliop(get ? get->upstream.lock() : nullptr);

    if(!get || !cliop) {
        us->getop = get = std::make_shared<GWGet>();
        get->upstream = cliop = us->upstream.get(us->usname)
                .autoExec(false)
                .syncCancel(false)
                .result([get, us](client::Result&& result)
                {
                    // on client worker
                    // 2. error prior to reExec()

                    std::string msg;
                    try {
                        result();
                        msg = "onInit() unexpected success/error";
                        log_err_printf(_logget, "'%s' GET cached onInit() unexpected success/error\n", us->usname.c_str());
                    } catch (std::exception& e) {
                        msg = e.what();
                        log_debug_printf(_logget, "'%s' GET cached init error: %s\n", us->usname.c_str(), e.what());
                    }

                    decltype (get->setups) setups;
                    decltype (get->ops) ops;
                    {
                        Guard G(us->lock);
                        get->state = GWGet::Error;
                        get->error = msg;

                        setups = std::move(get->setups);
                        ops = std::move(get->ops);
                    }

                    for(auto& setup : setups) {
                        setup->error(msg);
                    }
                    for(auto& op : ops) {
                        op->error(msg);
                    }
                })
                .onInit([get, us](const Value& prototype){
                    // on client worker
                    // 2. upstream connected and (proto)type definition is available

                    log_debug_printf(_logget, "'%s' GET cached typed\n", us->usname.c_str());

                    decltype (get->setups) setups;
                    {
                        Guard G(us->lock);
                        assert(get->state == GWGet::Connecting);
                        get->state = GWGet::Idle;
                        get->prototype = prototype;
                        setups = std::move(get->setups);
                    }
                    for(auto& setup : setups) {
                        try {
                            setup->connect(prototype);
                        }catch(std::exception& e){ // eg. this is where a bad pvRequest is discovered
                            log_debug_printf(_logget, "'%s' GET cached init server error: %s\n", us->usname.c_str(), e.what());
                            setup->error(e.what());
                        }
                    }
                })
                .exec();
    }

    switch(get->state) {
    case GWGet::Connecting:
        log_debug_printf(_logget, "'%s' GET init conn\n", ctrl->name().c_str());
        get->setups.push_back(ctrl);
        break;
    case GWGet::Idle:
    case GWGet::Exec:
        log_debug_printf(_logget, "'%s' GET init typed\n", ctrl->name().c_str());
        ctrl->connect(get->prototype);
        break;
    case GWGet::Error:
        ctrl->error(get->error);
        return;
    }

    ctrl->onGet([get, us, cliop](std::unique_ptr<server::ExecOp>&& sop){
        // on server worker
        // 3. downstream executes

        Guard G(us->lock);

        switch(get->state) {
        case GWGet::Connecting:
            // A logic error in the client code
            log_exc_printf(_logget, "'%s' GET exec before connect()\n", us->usname.c_str());
            break;

        case GWGet::Idle: {
            // need to exec
            auto delay = us->get_holdoff * 1e-3;
            auto now(epicsTime::getCurrent());
            auto age(now - get->lastget);

            if(get->firstget || age <= delay)
                delay = 0.0;

            log_debug_printf(_logget, "'%s' GET exec issue %.03f\n", us->usname.c_str(), delay);

            // avoid ref loop GWGet::delay -> client::Operation -> GWChan -> GWUpstream -> GWGet
            std::weak_ptr<GWGet> wget(get);
            std::weak_ptr<client::Operation> wcliop(cliop);
            std::weak_ptr<GWUpstream> wus(us);

            get->delay = sop->timerOneShot(delay, [wget, wus, wcliop](){
                // on server worker
                // 4. holdoff timer expires

                auto get(wget.lock());
                auto cliop(wcliop.lock());
                auto us(wus.lock());
                if(!get || !cliop || !us)
                    return; // all downstream disconnect/cancel while holdoff running, and timer somehow not canceled.  Probably can't happen.

                log_debug_printf(_logget, "'%s' GET holdoff expires\n", us->usname.c_str());

                cliop->reExecGet([get, us](client::Result&& result) {
                    // on client worker
                    // 5. upstream provides result


                    decltype (get->ops) ops;
                    {
                        Guard G(us->lock);
                        assert(get->state==GWGet::Exec);
                        get->state = GWGet::Idle;

                        ops = std::move(get->ops);
                    }

                    try {
                        auto value(result());
                        log_debug_printf(_logget, "'%s' GET exec complete\n", us->usname.c_str());
                        for(auto& op : ops) {
                            op->reply(value);
                        }
                    }catch(std::exception& e){
                        log_debug_printf(_logget, "'%s' GET exec complete err='%s'\n", us->usname.c_str(), e.what());
                        for(auto& op : ops) {
                            op->error(e.what());
                        }
                    }
                });

                // note time at which upstream GET is issued
                get->lastget = epicsTime::getCurrent();
                get->firstget = false;
            });

            get->state = GWGet::Exec;
            get->ops.emplace_back(std::move(sop));
            break;
        }
        case GWGet::Exec:
            // combine with in progress upstream GET
            log_debug_printf(_logget, "'%s' GET exec combine\n", us->usname.c_str());
            get->ops.emplace_back(std::move(sop));
            break;
        case GWGet::Error:
            log_debug_printf(_logget, "'%s' GET exec error: %s\n", us->usname.c_str(), get->error.c_str());
            sop->error(get->error);
            break;
        }
    });
}

static
void onGetPut(const std::shared_ptr<GWChan>& pv, const std::shared_ptr<server::ConnectOp>& ctrl)
{
    // on server worker
    // 1. downstream creating PUT or uncached GET operation

    log_debug_printf(_log, "'%s' GET/PUT init\n", ctrl->name().c_str()); // ============ GET/PUT

    auto result = [ctrl](client::Result&& result)
    {
        // on client worker
        // 2. error prior to reExec()

        // syncs client worker with server worker
        try {
            result();
            ctrl->error("onInit() unexpected success/error");
            log_err_printf(_log, "onInit() unexpected success/error%s", "!");
        } catch (std::exception& e) {
            ctrl->error(e.what());
            log_debug_printf(_log, "'%s' GET init error: %s\n", ctrl->name().c_str(), e.what());
        }
    };

    auto onInit = [ctrl, pv](const Value& prototype)
    {
        // on client worker
        // 2. upstream connected and (proto)type definition is available

        log_debug_printf(_log, "'%s' GET typed\n", ctrl->name().c_str());

        // syncs client worker with server worker
        ctrl->connect(prototype);
        // downstream may now execute
    };

    std::shared_ptr<client::Operation> cliop;

    // 1. Initiate operation
    if(ctrl->op()==server::ConnectOp::Get) {
        cliop = pv->us->upstream.get(pv->us->usname)
                     .autoExec(false)
                     .syncCancel(false)
                     .rawRequest(ctrl->pvRequest())
                     .result(std::move(result))
                     .onInit(std::move(onInit))
                     .exec();

    } else { // Put
        cliop = pv->us->upstream.put(pv->us->usname)
            .autoExec(false)
            .syncCancel(false)
            .rawRequest(ctrl->pvRequest()) // for PUT, always pass through w/o cache/dedup
            .result(std::move(result))
            .onInit(std::move(onInit))
            .exec();
    }

    // handles both plain CMD_GET as well as Get action on CMD_PUT
    ctrl->onGet([cliop](std::unique_ptr<server::ExecOp>&& sop){
        // on server worker
        // 3. downstream executes
        std::shared_ptr<server::ExecOp> op(std::move(sop));
        log_debug_printf(_log, "'%s' GET exec\n", op->name().c_str());

        // async request from server to client
        cliop->reExecGet([op](client::Result&& result) {
            // on client worker
            // 4. upstream execution complete

            log_debug_printf(_log, "'%s' GET exec done\n", op->name().c_str());

            // syncs client worker with server worker
            try {
                op->reply(result());
            } catch (std::exception& e) {
                op->error(e.what());
            }
        });
    });

    ctrl->onPut([cliop, pv](std::unique_ptr<server::ExecOp>&& sop, Value&& arg){
        // on server worker
        // 3. downstream executes
        std::shared_ptr<server::ExecOp> op(std::move(sop));

        bool permit = pv->allow_put;
        if(pv->audit) {
            AuditEvent evt{epicsTime::getCurrent(), pv->us->usname, op->name(), arg, op->credentials()};
            pv->us->src.auditPush(std::move(evt));
        }

        log_debug_printf(_log, "'%s' PUT exec%s\n", op->name().c_str(), permit ? "" : " DENY");

        if(!permit) {
            op->error("Put permission denied by gateway");
            return;
        }

        // async request from server to client
        cliop->reExecPut(arg, [op](client::Result&& result) {
            // on client worker
            // 4. upstream execution complete

            log_debug_printf(_log, "'%s' PUT exec done\n", op->name().c_str());

            // syncs client worker with server worker
            try {
                result();
                op->reply();
            } catch (std::exception& e) {
                op->error(e.what());
            }
        });
    });

    // just need to keep client op alive
    ctrl->onClose([cliop](const std::string&) {
        log_debug_printf(_log, "op close '%s'\n", cliop->name().c_str());
    });
}

void GWChan::onOp(const std::shared_ptr<GWChan>& pv, std::unique_ptr<server::ConnectOp>&& sop)
{
    // on server worker
    // 1. downstream creating operation

    std::shared_ptr<server::ConnectOp> ctrl(std::move(sop));

    auto pvReq(ctrl->pvRequest());
    bool docached = true;
    pvReq["record._options.cache"].as(docached);

    if(ctrl->op()==server::ConnectOp::Info) {
        onInfo(pv, ctrl);

    } else if(ctrl->op()==server::ConnectOp::Get && !docached && !pv->allow_uncached) {
        // GW config doesn't grant UNCACHED, and client has requested cache=false
        ctrl->error("Gateway disallows uncachable GET");

    } else if(ctrl->op()==server::ConnectOp::Get && docached) {
        onGetCached(pv, ctrl);

    } else if(ctrl->op()==server::ConnectOp::Get || ctrl->op()==server::ConnectOp::Put) {
        onGetPut(pv, ctrl);

    } else {
        ctrl->error(SB()<<"p4p.gw unsupported operation "<<ctrl->op());
    }
}

static
void onSubEvent(const std::shared_ptr<GWSubscription>& sub, const std::shared_ptr<GWChan>& pv)
{
    // on workQworker
    auto cli(sub->upstream.lock());
    if(!cli)
        return;

    log_debug_printf(_logmon, "'%s' MONITOR wakeup\n", cli->name().c_str());

    for(unsigned i=0; i<4u; i++) {
        try {
            auto val(cli->pop());
            if(!val)
                return; // queue emptied

            log_debug_printf(_logmon, "'%s' MONITOR event\n", cli->name().c_str());

            Guard G(pv->us->lock);
            if(!sub->current)
                sub->current = val; // first update
            else
                sub->current.assign(val); // accumulate deltas

            for(auto& ctrl : sub->controls)
                ctrl->post(val);

         } catch(client::Finished&) {
            log_debug_printf(_logmon, "'%s' MONITOR finish\n", cli->name().c_str());

            Guard G(pv->us->lock);
            pv->us->subscription.reset();
            for(auto& ctrl : sub->controls)
                ctrl->finish();

         } catch(std::exception& e) {
            log_warn_printf(_logmon, "'%s' MONITOR error: %s\n",
                            cli->name().c_str(), e.what());
         }
    }

    log_debug_printf(_logmon, "'%s' MONITOR resched\n", cli->name().c_str());

    // queue not empty, reschedule for later to give other subscriptions a chance
    pv->us->workQ->push([sub, pv](){ onSubEvent(sub, pv); });
}

void GWChan::onSubscribe(const std::shared_ptr<GWChan>& pv, std::unique_ptr<server::MonitorSetupOp>&& sop)
{
    // on server worker

    std::shared_ptr<server::MonitorSetupOp> op(std::move(sop));

    auto pvReq(op->pvRequest());
    auto docache = true;
    pvReq["record._options.cache"].as(docache);

    if(!docache && !pv->allow_uncached) {
        op->error("Gateway disallows uncachable monitor");
        return;
    }

    Guard G(pv->us->lock);

    auto sub(pv->us->subscription.lock());
    auto cli(sub ? sub->upstream.lock() : nullptr);
    if(sub && cli) {
        // re-use
        switch(sub->state) {
        case GWSubscription::Connecting:
            log_debug_printf(_logmon, "'%s' MONITOR init conn\n", op->name().c_str());
            sub->setups.push_back(op);
            goto done;

        case GWSubscription::Running: {
            log_debug_printf(_logmon, "'%s' MONITOR init run\n", op->name().c_str());
            auto ctrl(op->connect(sub->current));
            ctrl->post(sub->current); // post current as initial for new subscriber
            sub->controls.emplace_back(std::move(ctrl));
            goto done;
        }

        case GWSubscription::Error:
            break;
        }
    }

    log_debug_printf(_logmon, "'%s' MONITOR new\n", op->name().c_str());

    // start new subscription
    sub = std::make_shared<GWSubscription>();
    sub->setups.push_back(op);
    {
        auto req = pv->us->upstream.monitor(pv->us->usname)
                .syncCancel(false)
                .maskConnected(true) // upstream should already be connected
                .maskDisconnected(true); // handled by the client Connect op

        if(!docache)
            req.rawRequest(op->pvRequest()); // when not cached, pass through upstream client request verbatim.

        sub->upstream = cli = req
                .event([sub, pv](client::Subscription& cli)
        {
            // on client worker

            // only invoked if there is an early error.
            // replaced below for starting

            try {
                cli.pop(); // expected to throw
                throw std::runtime_error("not error??");
            }catch(std::exception& e){
                log_warn_printf(_logmon, "'%s' MONITOR setup error: %s\n", cli.name().c_str(), e.what());

                Guard G(pv->us->lock);
                pv->us->subscription.reset();
                sub->state = GWSubscription::Error;
                for(auto& op : sub->setups)
                    op->error(e.what());
            }
        })
                .onInit([sub, pv](client::Subscription& cli, const Value& prototype)
        {
            // on client worker

            log_debug_printf(_logmon, "'%s' MONITOR typed\n", cli.name().c_str());

            //auto clisub(cli.shared_from_this());

            cli.onEvent([sub, pv](client::Subscription& cli) { // replace earlier .event(...)
                // on client worker

                log_debug_printf(_logmon, "'%s' MONITOR wakeup\n", cli.name().c_str());

                pv->us->workQ->push([sub, pv]() { onSubEvent(sub, pv); });
            });

            // syncs client worker with server worker
            {
                Guard G(pv->us->lock);
                sub->state = GWSubscription::Running;
                auto setups(std::move(sub->setups));
                for(auto& setup : setups)
                    sub->controls.push_back(setup->connect(prototype));
            }
        })
                .exec();
    }

    if(docache)
        pv->us->subscription = sub;

    // BUG: need to unlock before onClose()
done:
    // tie client subscription lifetime (and by extension GWSubscription) to server op.
    // Reference to CLI stored in internal server OP struct, so no ref. loop
    op->onClose([cli](const std::string&) {
        log_debug_printf(_log, "sub close '%s'\n", cli->name().c_str());
    });
}

void GWSource::onCreate(std::unique_ptr<server::ChannelControl> &&op)
{
    // on server worker

    // Server worker may make synchronous calls to client worker.
    // To avoid deadlock, client worker must not make synchronous calls to server worker

    // Server operation handles may hold strong references to client operation handles
    // To avoid a reference loop, client operation handles must not hold strong refs.
    // to server handles.

    std::shared_ptr<GWChan> pv;
    {
        PyLock L;

        pv = GWProvider_makeChannel(this, &op);
    }

    if(!pv) {
        return; // not our PV.  Let other GWSource s try.

    } else if(!pv->us->connector->connected()) {
        // ours, but something went wrong.
        log_debug_printf(_log, "%p makeChannel returned '%s'\n", this,
                         op ? op->name().c_str() : "dead channel");
        if(op)
            op->close();
        return;
    }

    assert(pv->dschannel);
    auto& ctrl = pv->dschannel;

    ctrl->updateInfo(pv->reportInfo);

    ctrl->onRPC([pv](std::unique_ptr<server::ExecOp>&& op, Value&& arg) mutable {
        // on server worker
        GWChan::onRPC(pv, std::move(op), std::move(arg));
    });

    ctrl->onOp([pv](std::unique_ptr<server::ConnectOp>&& sop) mutable { // INFO/GET/PUT
        // on server worker
        GWChan::onOp(pv, std::move(sop));
    }); // onOp

    ctrl->onSubscribe([pv](std::unique_ptr<server::MonitorSetupOp>&& sop) mutable {
        GWChan::onSubscribe(pv, std::move(sop));
    }); // onSubscribe

    log_debug_printf(_log, "%p onCreate '%s' as '%s' success\n", this, pv->dsname.c_str(), pv->us->usname.c_str());
}

GWSearchResult GWSource::test(const std::string &usname)
{
    Guard G(mutex);

    auto it(channels.find(usname));

    log_debug_printf(_log, "%p '%s' channel cache %s\n", this, usname.c_str(),
                     (it==channels.end()) ? "miss" : "hit");
    if(it==channels.end()) {

        auto chan(std::make_shared<GWUpstream>(usname, *this));

        auto pair = channels.insert(std::make_pair(usname, chan));
        assert(pair.second); // we already checked
        it = pair.first;

        log_debug_printf(_log, "%p new upstream channel '%s'\n", this, usname.c_str());
    }

    if(it->second->gcmark) {
        log_debug_printf(_log, "%p unmark '%s'\n", this, usname.c_str());
    }
    it->second->gcmark = false;
    auto usconn = it->second->connector->connected();

    log_debug_printf(_log, "%p test '%s' -> %c\n", this, usname.c_str(), usconn ? '!' : '_');

    return usconn ? GWSearchClaim : GWSearchIgnore;
}



std::shared_ptr<GWChan> GWSource::connect(const std::string& dsname,
                                          const std::string& usname,
                                          std::unique_ptr<server::ChannelControl> *ctrl)
{
    std::shared_ptr<GWChan> ret;

    Guard G(mutex);

    auto it(channels.find(usname));
    if(it!=channels.end() && it->second->connector->connected()) {
        std::shared_ptr<server::ChannelControl> op(std::move(*ctrl));
        ret.reset(new GWChan(usname, dsname, it->second, op));
    }

    log_debug_printf(_log, "%p connect '%s' as '%s' -> %c\n", this, usname.c_str(), dsname.c_str(), ret ? '!' : '_');

    return ret;
}

void GWSource::sweep()
{
    log_debug_printf(_log, "%p sweeps\n", this);

    std::vector<std::shared_ptr<GWUpstream>> trash;
    // garbage disposal after unlock
    Guard G(mutex);

    {
        auto it(channels.begin()), end(channels.end());
        while(it!=end) {
            auto cur(it++);

            if(cur->second.use_count() > 1u) {
                // no-op

            } else if(!cur->second->gcmark) {
                log_debug_printf(_log, "%p marked '%s'\n", this, cur->first.c_str());
                cur->second->gcmark = true;

            } else { // one for GWSource::channels map
                log_debug_printf(_log, "%p swept '%s'\n", this, cur->first.c_str());
                trash.emplace_back(std::move(cur->second));
                upstream.cacheClear(cur->first);
                channels.erase(cur);
            }
        }
    }
}

void GWSource::disconnect(const std::string& usname) {}
void GWSource::forceBan(const std::string& host, const std::string& usname) {}
void GWSource::clearBan() {}

void GWSource::cachePeek(std::set<std::string> &names) const {}

void GWSource::auditPush(AuditEvent&& revt)
{
    auto evt(std::move(revt));
    {
        constexpr size_t limit = 100u; // TODO: configurable?

        Guard G(mutex);

        if(audits.size() == limit)
            evt.usname.clear(); // overflow

        if(audits.size() <= limit)
            audits.push_back(std::move(evt));

        if(audits.size() > 1u)
            return; // already scheduled
    }

    workQ->push([this]() {
        // on queue worker

        decltype (audits) todo;
        {
            Guard G(mutex);
            todo = std::move(audits);
        }

        std::list<std::string> msgs;

        for(auto& audit : todo) {
            std::ostringstream strm;

            // log line format
            //  <timestamp> ' ' [ <method> '/' <account> ] '@' <peer> ' ' <dsname> " as " <usname> [ " -> " <value> ]
            //  <timestamp> " ... put audit log overflow"
            {
                char buf[64];
                audit.now.strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S.%09f");
                strm<<buf<<' ';
            }

            if(audit.usname.empty()) {
                strm<<"... put audit log overflow";

            } else {
                if(audit.cred) {
                    strm<<audit.cred->method<<'/'<<audit.cred->account;
                }
                strm<<'@'<<audit.cred->peer<<' '<<audit.dsname<<" as "<<audit.usname;

                if(auto val = audit.val["value"]) {
                    if(val.type().kind()!=Kind::Compound)
                        strm<<" -> "<<val.format().arrayLimit(10u);
                }
            }

            msgs.push_back(strm.str());
        }

        GWProvider_audit(this, msgs);
    });
}

void GWSource::run()
{
    while(auto work = workQ->pop()) {
        if(!work)
            break; // NULL means stop

        try {
            work();
        }catch(std::exception &e) {
            log_exc_printf(_log, "Unhandled exception from workQ: %s : %s\n",
                           work.target_type().name(), e.what());
        }
    }
}

} // namespace p4p
