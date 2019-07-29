
#include "gwchannel.h"
#include "_gw.h"

typedef epicsGuard<epicsMutex> Guard;
typedef epicsGuardRelease<epicsMutex> UnGuard;

size_t GWProvider::num_instances;
size_t GWChan::num_instances;
size_t GWChan::Requester::num_instances;
size_t GWMon::num_instances;
size_t GWMon::Requester::num_instances;
size_t ProxyPut::num_instances;
size_t ProxyPut::Requester::num_instances;
size_t ProxyRPC::num_instances;
size_t ProxyRPC::Requester::num_instances;


GWChan::Requester::Requester()
    :poked(true)
{
    REFTRACE_INCREMENT(num_instances);
}
GWChan::Requester::~Requester() {
    REFTRACE_DECREMENT(num_instances);
}

void GWChan::Requester::latch(strong_t& chans)
{
    chans.clear();
    chans.reserve(ds_requesters.size());
    for(weak_t::const_iterator it(ds_requesters.begin()), end(ds_requesters.end()); it!=end; ++it)
    {
        GWChan::shared_pointer R(it->second.lock());
        if(!R) continue;
        chans.push_back(GWChan::shared_pointer());
        chans.back().swap(R);
    }
}

std::string GWChan::Requester::getRequesterName() { return "GWChan::Requester"; } // not sure where this would appear

void GWChan::Requester::message(std::string const & message,pva::MessageType messageType)
{
    strong_t chans;
    {
        Guard G(mutex);
        latch(chans);
    }
    for(size_t i=0, N=chans.size(); i<N; i++) {
        chans[i]->message(message, messageType);
    }
}

void GWChan::Requester::channelCreated(const pvd::Status& status, pva::Channel::shared_pointer const & channel)
{
    strong_t chans;
    {
        Guard G(mutex);
        latch(chans);
    }
    TRACE(chans.size());
    for(size_t i=0, N=chans.size(); i<N; i++) {
        pva::ChannelRequester::shared_pointer req(chans[i]->ds_requester.lock());
        if(req)
            req->channelCreated(status, chans[i]);
    }
}

void GWChan::Requester::channelStateChange(pva::Channel::shared_pointer const & channel, pva::Channel::ConnectionState connectionState)
{
    strong_t chans;
    {
        Guard G(mutex);
        latch(chans);
    }
    TRACE(chans.size());
    for(size_t i=0, N=chans.size(); i<N; i++) {
        pva::ChannelRequester::shared_pointer req(chans[i]->ds_requester.lock());
        if(req)
            req->channelStateChange(chans[i], connectionState);
    }
}

GWChan::GWChan(const std::tr1::shared_ptr<GWProvider>& provider,
               const std::string& name,
               const pva::ChannelRequester::weak_pointer& requester)
    :name(name)
    ,provider(provider)
    ,ds_requester(requester)
    ,allow_put(false)
    ,allow_rpc(false)
    ,allow_uncached(false)
{
    REFTRACE_INCREMENT(num_instances);
}
GWChan::~GWChan() {
    destroy();
    REFTRACE_DECREMENT(num_instances);
}

void GWChan::destroy() {
    Guard G(us_requester->mutex);
    us_requester->ds_requesters.erase(this);
}

std::tr1::shared_ptr<pva::ChannelProvider> GWChan::getProvider() { return provider.lock(); }
std::string GWChan::getRemoteAddress() { return us_channel->getRequesterName(); }
std::string GWChan::getChannelName() { return name; }
std::tr1::shared_ptr<pva::ChannelRequester> GWChan::getChannelRequester() { return ds_requester.lock(); }

void GWChan::disconnect()
{
    destroy();
    us_requester->channelStateChange(shared_from_this(), pva::Channel::DESTROYED);
}

void GWChan::getField(pva::GetFieldRequester::shared_pointer const & requester,std::string const & subField)
{
    return us_channel->getField(requester, subField);
}

pva::ChannelGet::shared_pointer GWChan::createChannelGet(
        pva::ChannelGetRequester::shared_pointer const & requester,
        pvd::PVStructure::shared_pointer const & pvRequest)
{
    TRACE("");
    return us_channel->createChannelGet(requester, pvRequest);
}

GWMon::Requester::Requester(const std::string &usname)
    :name(usname)
{
    REFTRACE_INCREMENT(num_instances);
}
GWMon::Requester::~Requester() {
    REFTRACE_DECREMENT(num_instances);
}

void GWMon::Requester::latch(strong_t& mons)
{
    mons.clear();
    mons.reserve(ds_ops.size());
    for(weak_t::const_iterator it(ds_ops.begin()), end(ds_ops.end()); it!=end; ++it)
    {
        GWMon::shared_pointer R(it->second.lock());
        if(!R) continue;
        mons.push_back(GWMon::shared_pointer());
        mons.back().swap(R);
    }
}

std::string GWMon::Requester::getRequesterName() {return "GWMon::Requester"; } // not sure where this would appear

void GWMon::Requester::channelDisconnect(bool destroy)
{
    strong_t mons;
    {
        Guard G(mutex);
        latch(mons);
    }
    TRACE(mons.size());
    for(size_t i=0, N=mons.size(); i<N; i++) {
        mons[i]->close();
        mons[i]->notify();
    }
}

void GWMon::Requester::monitorConnect(pvd::Status const & status,
                                      pva::MonitorPtr const & monitor, pvd::StructureConstPtr const & structure)
{
    pvd::PVStructurePtr container;
    if(structure)
        container = pvd::getPVDataCreate()->createPVStructure(structure);

    if(status.isSuccess() && monitor)
        (void)monitor->start();

    strong_t mons;
    {
        Guard G(mutex);
        latch(mons);
        valid.clear();
        if(status.isSuccess() && container) {
            complete = container;
        } else {
            TRACE(status<<" no Initial?!?");
            complete.reset();
            return;
        }
    }
    TRACE(status<<" "<<mons.size()<<" "<<status);
    for(size_t i=0, N=mons.size(); i<N; i++) {
        mons[i]->open(structure);
        mons[i]->notify();
    }
}

void GWMon::Requester::monitorEvent(pva::MonitorPtr const & monitor)
{
    TRACE("");
    strong_t mons;
    {
        Guard G(mutex);
        latch(mons);
    }
    TRACE(mons.size());

    for(pvd::MonitorElement::Ref it(monitor); it; ++it)
    {
        pva::MonitorElement& elem(*it);

        for(size_t i=0, N=mons.size(); i<N; i++) {
            mons[i]->post(*elem.pvStructurePtr, *elem.changedBitSet, *elem.overrunBitSet);
        }

        if(complete) {
            assert(complete->getStructure()==elem.pvStructurePtr->getStructure());

            complete->copyUnchecked(*elem.pvStructurePtr,
                                    *elem.changedBitSet);
            valid |= *elem.changedBitSet;
        }
    }

    for(size_t i=0, N=mons.size(); i<N; i++) {
        mons[i]->notify();
    }
}

void GWMon::Requester::unlisten(pva::MonitorPtr const & monitor)
{
    TRACE("");
    strong_t mons;
    {
        Guard G(mutex);
        latch(mons);
    }
    TRACE(mons.size());
    for(size_t i=0, N=mons.size(); i<N; i++) {
        mons[i]->finish();
        mons[i]->notify();
    }
}

GWMon::GWMon(const std::string &name,
             const pva::MonitorRequester::shared_pointer& requester,
             const pvd::PVStructure::const_shared_pointer &pvRequest,
             const pva::MonitorFIFO::Source::shared_pointer& source,
             pva::MonitorFIFO::Config *conf)
    :pva::MonitorFIFO(requester, pvRequest, source, conf)
    ,name(name)
{
    REFTRACE_INCREMENT(num_instances);
}

GWMon::~GWMon()
{
    Guard G(us_requester->mutex);
    us_requester->ds_ops.erase(this);
    REFTRACE_DECREMENT(num_instances);
}

pva::Monitor::shared_pointer GWChan::createMonitor(pva::MonitorRequester::shared_pointer const & requester,
                                                   pvd::PVStructure::shared_pointer const & pvRequest)
{
    GWProvider::shared_pointer provider(this->provider.lock());
    if(!provider) {
        requester->monitorConnect(pvd::Status::error("Dead Provider"),
                                  pva::MonitorPtr(),
                                  pvd::StructureConstPtr());
        return pva::MonitorPtr();
    }

    bool cache = true;
    // client specifically requesting uncached
    pvd::PVScalar::const_shared_pointer V = pvRequest->getSubField<pvd::PVScalar>("record._options.cache");
    if(V) {
        try {
            cache &= V->getAs<pvd::boolean>();
        }catch(std::runtime_error&){
            // treat typo as non-cache
            cache = false;
        }
    }

    if(!cache) {
        if(!allow_uncached) {
            TRACE("ERROR uncached");
            requester->monitorConnect(pvd::Status::error("Gateway disallows uncachable monitor"), pvd::MonitorPtr(), pvd::StructureConstPtr());
            return pvd::MonitorPtr();
        } else {
            TRACE("ALLOW uncached");
            return us_channel->createMonitor(requester, pvRequest);
        }
    }

    // build upstream request
    // TODO: allow some downstream requests?  (queueSize?)
    pvd::PVStructurePtr up(pvd::ValueBuilder()
                           .addNested("fields")
                           .endNested()
                           .buildPVStructure());

    // create cache key.
    // use non-aliased upstream channel name
    std::string key, usname(us_channel->getChannelName());
    {
        std::ostringstream strm;
        strm<<usname<<"\n"<<(*up);
        key = strm.str();
    }

    GWMon::shared_pointer ret(new GWMon(name, requester, pvRequest));
    ret->channel = shared_from_this();

    GWMon::Requester::shared_pointer entry;
    bool create;
    {
        Guard G(provider->mutex);

        GWProvider::monitors_t::iterator it(provider->monitors.find(key));

        if(it!=provider->monitors.end()) {
            entry = it->second.lock();
        }

        create = !entry;
        if(create) {
            entry.reset(new GWMon::Requester(usname));
            provider->monitors[key] = entry;
        }
    }

    pvd::PVStructurePtr initial;
    pvd::BitSet ivalid;
    {
        Guard G(entry->mutex);
        entry->ds_ops[ret.get()] = ret;
        ret->us_requester = entry;

        if(create) {
            entry->us_op = us_channel->createMonitor(entry, up);
        }
        if(entry->complete) {
            // upstream already connected
            // TODO: MonitorFIFO::open() avoid message()?  we could post() w/o copy
            initial = pvd::getPVDataCreate()->createPVStructure(entry->complete);
            ivalid = entry->valid;
        }
    }
    if(initial) {
        ret->open(initial->getStructure());
        ret->post(*initial, ivalid);
        ret->notify();
    }

    TRACE("CREATE cached "<<(create?'T':'F')<<" "<<(initial?'T':'F'));
    return ret;
}

ProxyPut::Requester::Requester(const requester_type::weak_pointer& downstream)
    :downstream(downstream)
{
    REFTRACE_INCREMENT(num_instances);
}
ProxyPut::Requester::~Requester() {
    REFTRACE_DECREMENT(num_instances);
}

std::string ProxyPut::Requester::getRequesterName() {
    requester_type::shared_pointer ds(downstream.lock());
    return ds ? ds->getRequesterName() : std::string();
}
void ProxyPut::Requester::message(std::string const & message,pva::MessageType messageType) {
    requester_type::shared_pointer ds(downstream.lock());
    return ds ? ds->message(message, messageType) : requester_type::message(message, messageType);
}

void ProxyPut::Requester::channelDisconnect(bool destroy) {
    requester_type::shared_pointer ds(downstream.lock());
    if(ds) ds->channelDisconnect(destroy);
}

void ProxyPut::Requester::channelPutConnect(
        const epics::pvData::Status& status,
        ChannelPut::shared_pointer const & channelPut,
        epics::pvData::Structure::const_shared_pointer const & structure)
{
    TRACE(status);
    pvd::Status err(status);
    requester_type::shared_pointer ds(downstream.lock());
    gw_operation_type::shared_pointer op(operation.lock());
    if(!ds) return;

    if(!op)
        err = pvd::Status::error("Dead channel");
    ds->channelPutConnect(status, op, structure);
}

void ProxyPut::Requester::putDone(
        const epics::pvData::Status& status,
        ChannelPut::shared_pointer const & channelPut)
{
    TRACE(status);
    pvd::Status err(status);
    requester_type::shared_pointer ds(downstream.lock());
    gw_operation_type::shared_pointer op(operation.lock());
    if(!ds) return;

    if(!op)
        err = pvd::Status::error("Dead channel");
    ds->putDone(err, op);
}

void ProxyPut::Requester::getDone(
        const epics::pvData::Status& status,
        ChannelPut::shared_pointer const & channelPut,
        epics::pvData::PVStructure::shared_pointer const & pvStructure,
        epics::pvData::BitSet::shared_pointer const & bitSet)
{
    TRACE(status);
    pvd::Status err(status);
    requester_type::shared_pointer ds(downstream.lock());
    gw_operation_type::shared_pointer op(operation.lock());
    if(!ds) return;

    if(!op)
        err = pvd::Status::error("Dead channel");
    ds->getDone(err, op, pvStructure, bitSet);
}


ProxyPut::ProxyPut(const std::tr1::shared_ptr<struct GWChan>& channel,
                   const requester_type::shared_pointer& requester)
    :channel(channel)
    ,us_requester(new Requester(requester))
{
    REFTRACE_INCREMENT(num_instances);
}

ProxyPut::~ProxyPut() {
    REFTRACE_DECREMENT(num_instances);
}

void ProxyPut::put(
        epics::pvData::PVStructure::shared_pointer const & pvPutStructure,
        epics::pvData::BitSet::shared_pointer const & putBitSet)
{
    TRACE(channel->allow_put);
    if(channel->allow_put) {
        pva::ChannelPut::shared_pointer op;
        {
            Guard G(mutex);
            op = us_op;
        }
        if(channel->audit) {
            std::ostringstream strm;
            {
                char buf[64];
                epicsTime::getCurrent().strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S.%09f");
                strm<<buf<<' ';
            }
            pva::ChannelRequester::shared_pointer req(channel->ds_requester);
            if(req) {
                pva::PeerInfo::const_shared_pointer info(req->getPeerInfo());
                if(info) {
                    if(info->identified)
                        strm<<info->authority<<'/'<<info->account;
                    strm<<'@'<<info->peer<<' ';
                } else {
                    strm<<req->getRequesterName()<<' ';
                }
            } else {
                strm<<"??? ";
            }
            strm<<channel->name<<" as "<<channel->us_channel->getChannelName();
            pvd::PVScalar::const_shared_pointer value(pvPutStructure->getSubField<pvd::PVScalar>("value"));
            if(value) {
                strm<<" -> "<<value->getAs<std::string>();
            }

            GWProvider::audit_log_t entry(1);
            entry.back() = strm.str();

            GWProvider::shared_pointer provider(channel->provider);
            bool poke_audit = false;
            if(provider) {
                {
                    Guard G(provider->mutex);

                    poke_audit = provider->audit_log.empty();

                    if(provider->audit_log.size()<100u) {
                        provider->audit_log.splice(provider->audit_log.end(), entry);

                    } else if(provider->audit_log.size()==100u) {
                        provider->audit_log.push_back("... put audit log overflow");

                    } else {
                        // in overflow
                    }
                }
                if(poke_audit)
                    provider->audit_wakeup.signal();
            }
        }
        if(op) // possible race with createChannelPut() if we get called recursively (PVA client doesn't do this)
            op->put(pvPutStructure, putBitSet);
    } else if(requester_type::shared_pointer req = us_requester->downstream.lock()) {
        req->putDone(pvd::Status::error("Put not permitted"), shared_from_this());
    }
}

pva::ChannelPut::shared_pointer
GWChan::createChannelPut(
        pva::ChannelPutRequester::shared_pointer const & requester,
        pvd::PVStructure::shared_pointer const & pvRequest)
{
    TRACE("");
    std::tr1::shared_ptr<ProxyPut> op(new ProxyPut(shared_from_this(), requester));
    op->us_requester->operation = op; // store as weak_ptr

    pva::ChannelPut::shared_pointer base(us_channel->createChannelPut(op->us_requester, pvRequest));

    Guard G(op->mutex);
    op->us_op = base;
    return op;
}


ProxyRPC::Requester::Requester(const requester_type::weak_pointer& downstream)
    :downstream(downstream)
{
    REFTRACE_INCREMENT(num_instances);
}
ProxyRPC::Requester::~Requester() {
    REFTRACE_DECREMENT(num_instances);
}

std::string ProxyRPC::Requester::getRequesterName() {
    requester_type::shared_pointer ds(downstream.lock());
    return ds ? ds->getRequesterName() : std::string();
}
void ProxyRPC::Requester::message(std::string const & message,pva::MessageType messageType) {
    requester_type::shared_pointer ds(downstream.lock());
    return ds ? ds->message(message, messageType) : requester_type::message(message, messageType);
}

void ProxyRPC::Requester::channelDisconnect(bool destroy) {
    requester_type::shared_pointer ds(downstream.lock());
    if(ds) ds->channelDisconnect(destroy);
}

void ProxyRPC::Requester::channelRPCConnect(
        const epics::pvData::Status& status,
        ChannelRPC::shared_pointer const & rpc)
{
    TRACE(status);
    pvd::Status err(status);
    requester_type::shared_pointer ds(downstream.lock());
    gw_operation_type::shared_pointer op(operation.lock());
    if(!ds) return;

    if(!op)
        err = pvd::Status::error("Dead channel");
    ds->channelRPCConnect(status, op);

}

void ProxyRPC::Requester::requestDone(
        const epics::pvData::Status& status,
        ChannelRPC::shared_pointer const & rpc,
        epics::pvData::PVStructure::shared_pointer const & pvResponse)
{
    TRACE(status);
    pvd::Status err(status);
    requester_type::shared_pointer ds(downstream.lock());
    gw_operation_type::shared_pointer op(operation.lock());
    if(!ds) return;

    if(!op)
        err = pvd::Status::error("Dead channel");
    ds->requestDone(err, op, pvResponse);
}

ProxyRPC::ProxyRPC(const std::tr1::shared_ptr<struct GWChan>& channel,
                   const requester_type::shared_pointer& requester)
    :channel(channel)
    ,us_requester(new Requester(requester))
{
    REFTRACE_INCREMENT(num_instances);
}
ProxyRPC::~ProxyRPC() {
    REFTRACE_DECREMENT(num_instances);
}

void ProxyRPC::request(epics::pvData::PVStructure::shared_pointer const & pvArgument)
{
    TRACE(channel->allow_rpc);
    if(channel->allow_rpc) {
        pva::ChannelRPC::shared_pointer op;
        {
            Guard G(mutex);
            op = us_op;
        }
        op->request(pvArgument);
    } else if(requester_type::shared_pointer req = us_requester->downstream.lock()) {
        req->requestDone(pvd::Status::error("RPC not permitted"), shared_from_this(), pvd::PVStructurePtr());
    }
}

pva::ChannelRPC::shared_pointer
GWChan::createChannelRPC(
        pva::ChannelRPCRequester::shared_pointer const & requester,
        pvd::PVStructure::shared_pointer const & pvRequest)
{
    TRACE("");
    std::tr1::shared_ptr<ProxyRPC> op(new ProxyRPC(shared_from_this(), requester));
    op->us_requester->operation = op; // store as weak_ptr

    pva::ChannelRPC::shared_pointer base(us_channel->createChannelRPC(op->us_requester, pvRequest));

    Guard G(op->mutex);
    op->us_op = base;
    return op;
}

pva::ChannelProvider::shared_pointer GWProvider::buildClient(const std::string& name,const pva::Configuration::shared_pointer& conf)
{
    return pva::ChannelProviderRegistry::clients()->createProvider(name, conf);
}

std::tr1::shared_ptr<GWProvider> GWProvider::build(const std::string& name,
                                                   const pva::ChannelProvider::shared_pointer& provider)
{
    std::tr1::shared_ptr<GWProvider> ret(new GWProvider(name, provider));
    ret->dummyFind = pva::ChannelFind::buildDummy(ret);
    if(!pva::ChannelProviderRegistry::servers()->addSingleton(ret, false))
        throw std::runtime_error("Duplicate GW provider name");
    if(!ret.unique())
        throw std::logic_error("Created provider has ref.loop");
    return ret;
}

GWProvider::GWProvider(const std::string& name,
                       const pva::ChannelProvider::shared_pointer& provider)
    :name(name)
    ,client(provider)
    ,prevtime(epicsTime::getCurrent())
    ,audit_run(true)
    ,audit_runner(pvd::Thread::Config(this, &GWProvider::runAudit)
                  .name("GW Auditor")
                  .autostart(true))
    ,handle(0)
{
    REFTRACE_INCREMENT(num_instances);
    TRACE("");
}

GWProvider::~GWProvider() {
    TRACE("");
    {
        Guard G(mutex);
        audit_run = false;
    }
    audit_wakeup.signal();
    audit_holdoff.signal();
    audit_runner.exitWait();

    GWProvider_cleanup(this);
    REFTRACE_DECREMENT(num_instances);
}

GWSearchResult GWProvider::test(const std::string& usname)
{
    bool connected;

    Guard G(mutex);

    channels_t::iterator it(channels.find(usname));

    if(it==channels.end()) {
        connected = false;

        GWChan::Requester::shared_pointer req(new GWChan::Requester);
        pva::Channel::shared_pointer ch;

        channels[usname] = req;

        {
            //UnGuard U(G);
            ch = client->createChannel(usname, req);
        }

        // safe because us_channel only accessed through GWChan methods, and not indirectly
        // through GWChan::Requester methods (which could now be happening)
        req->us_channel = ch;
        TRACE("Add Channel Cache entry "<<usname);

    } else if(!it->second->us_channel) {
        connected = false;

    } else {
        it->second->poked = true;
        connected = it->second->us_channel->isConnected();
    }

    TRACE(usname<<" "<<(connected?'T':'F'));
    return connected ? GWSearchClaim : GWSearchIgnore;
}

GWChan::shared_pointer
GWProvider::connect(const std::string& dsname, const std::string &usname,
                    const epics::pvAccess::ChannelRequester::shared_pointer &requester)
{
    GWChan::Requester::shared_pointer us_requester;
    {
        Guard G(mutex);

        channels_t::iterator it(channels.find(usname));
        // Polling for connected is racy, but trying to track this in GWChan::Requester
        // is also as mulitple threads are involved (eg. client circuit send and recv workers).
        // We are likely being called from server recv worker.
        if(it!=channels.end() && it->second->us_channel && it->second->us_channel->isConnected())
            us_requester = it->second;
    }

    GWChan::shared_pointer ret;

    if(us_requester)
    {
        ret.reset(new GWChan(shared_from_this(), dsname, requester));
        ret->us_requester = us_requester;
        ret->us_channel = us_requester->us_channel;

        {
            Guard G(us_requester->mutex);

            us_requester->ds_requesters[ret.get()] = ret;
        }

        requester->channelCreated(pvd::Status(), ret);
    }
    if(!ret)
        throw std::runtime_error("Unable to connect");

    TRACE(ret);
    return ret;
}

void GWProvider::destroy() {
    TRACE("GWProvider");
}

std::string GWProvider::getProviderName() { return name; }

pva::ChannelFind::shared_pointer GWProvider::channelFind(std::string const & name,
                                                         pva::ChannelFindRequester::shared_pointer const & requester)
{
    GWSearchResult result = GWSearchClaim;
    pvd::Status sts;
    pva::PeerInfo::const_shared_pointer peer(requester->getPeerInfo());
    std::string peerHost;

    // Test negative result cache
    {
        Guard G(mutex);
        if(banPV.find(name)!=banPV.end()) {
            result = GWSearchIgnore;
        } else {
            if(peer)
                peerHost = peer->peer.substr(0, peer->peer.find_first_of(':'));

            if(banHost.find(peerHost)!=banHost.end()
                    || banHostPV.find(std::make_pair(peerHost, name))!=banHostPV.end())
                result = GWSearchIgnore;
        }
        if(result!=GWSearchClaim)
            TRACE("Ignore Banned "<<name<<" from "<<peerHost<<" "<<result);
    }

    if(result==GWSearchClaim) {
        TRACE("Check "<<name<<" for "<<peerHost);
        result = (GWSearchResult)GWProvider_testChannel(this, name.c_str(), peer ? peer->peer.c_str() : "");
    }

    if(result>GWSearchClaim) {
        Guard G(mutex);
        if(result==GWSearchBanPV) {
            TRACE("Ban PV "<<name);
            banPV.insert(name);
        } else if(result==GWSearchBanHost) {
            TRACE("Ban Host "<<peerHost);
            banHost.insert(peerHost);
        } else if(result==GWSearchBanHostPV) {
            TRACE("Ban Host+PV "<<peerHost<<" "<<name);
            banHostPV.insert(std::make_pair(peerHost, name));
        }
    }

    TRACE(name<<" "<<result);
    requester->channelFindResult(sts, dummyFind, result==GWSearchClaim);
    return dummyFind;
}

pva::ChannelFind::shared_pointer GWProvider::channelList(pva::ChannelListRequester::shared_pointer const & requester)
{
    requester->channelListResult(pvd::Status(),
                                 dummyFind,
                                 empty,
                                 true);
    return dummyFind;
}

pva::Channel::shared_pointer GWProvider::createChannel(std::string const & name,
                                                       pva::ChannelRequester::shared_pointer const & requester,
                                                       short priority, std::string const & address)
{
    pvd::Status sts;
    GWChan::shared_pointer ret(GWProvider_makeChannel(this, name, requester));

    if(!ret) {
        sts = pvd::Status::error("No such channel");
        requester->channelCreated(sts, ret);
    }
    // else the upstream channel will call channelCreated()
    TRACE(sts<<" "<<ret);
    return ret;
}

void GWProvider::sweep()
{
    std::vector<GWChan::Requester::shared_pointer> garbage;
    {
        Guard G(mutex);

        {
            channels_t::iterator it(channels.begin()), end(channels.end());
            while(it!=end) {
                channels_t::iterator cur(it++);
                if(cur->second->poked) {
                    cur->second->poked = false;

                } else if(cur->second.unique()) {
                    TRACE("Channel Cache discards "<<cur->second->us_channel->getChannelName());
                    garbage.push_back(cur->second);
                    channels.erase(cur);
                }
            }
        }

        {
            monitors_t::iterator it(monitors.begin()), end(monitors.end());
            while(it!=end) {
                monitors_t::iterator cur(it++);
                if(cur->second.expired())
                    monitors.erase(cur);
            }
        }
    }
}


void GWProvider::disconnect(const std::string& usname)
{
    GWChan::Requester::shared_pointer req;
    {
        Guard G(mutex);
        channels_t::iterator it(channels.find(usname));
        if(it!=channels.end()) {
            req = it->second;
            channels.erase(it);
        }
    }
    if(req) {
        req->us_channel->destroy();
    }
}

void GWProvider::forceBan(const std::string& host, const std::string& usname)
{
    Guard G(mutex);

    if(!host.empty() && !usname.empty()) {
        banHostPV.insert(std::make_pair(host, usname));

    } else if(!host.empty()) {
        banHost.insert(host);

    } else if(!usname.empty()) {
        banPV.insert(usname);
    }
}

void GWProvider::clearBan()
{
    Guard G(mutex);
    banHost.clear();
    banPV.clear();
    banHostPV.clear();
}

void GWProvider::cachePeek(std::set<std::string>& names) const
{
    names.clear();
    Guard G(mutex);

    for(channels_t::const_iterator it(channels.begin()), end(channels.end()); it!=end; ++it)
        names.insert(it->first);
}

void GWProvider::stats(GWStats& stats) const
{
    Guard G(mutex);

    stats.ccacheSize = channels.size();
    stats.mcacheSize = monitors.size();
    stats.banHostSize = banHost.size();
    stats.banPVSize = banPV.size();
    stats.banHostPVSize = banHostPV.size();
}

namespace {
void stats_add(GWProvider::ReportItem& ret, const pva::NetStats::Stats& lhs, const pva::NetStats::Stats& rhs, double period) {
    ret.operationTX = (lhs.operationBytes.tx - rhs.operationBytes.tx)/period;
    ret.operationRX = (lhs.operationBytes.rx - rhs.operationBytes.rx)/period;
    ret.transportTX = (lhs.transportBytes.tx - rhs.transportBytes.tx)/period;
    ret.transportRX = (lhs.transportBytes.rx - rhs.transportBytes.rx)/period;
}
}

void GWProvider::report(report_t& us, report_t& ds, double& period)
{
    us.clear();
    ds.clear();
    epicsTime now(epicsTime::getCurrent());

    std::vector<GWMon::Requester::shared_pointer> mons;
    // latch a copy of the presently active subscriptions
    {
        Guard G(mutex);
        mons.reserve(monitors.size());
        for(monitors_t::const_iterator it(monitors.begin()), end(monitors.end()); it!=end; ++it)
        {
            GWMon::Requester::shared_pointer M(it->second.lock());
            if(M)
                mons.push_back(M);
        }
        period = now - prevtime;
        prevtime = now;
    }

    us.reserve(mons.size());
    ds.reserve(mons.size()*2u); // assume ~# downstream per upstream on average

    for(size_t i=0; i<mons.size(); i++)
    {
        ReportItem ent;
        ent.usname = mons[i]->name; // same for this US, and all its DS

        const pva::NetStats* stats = dynamic_cast<const pva::NetStats*>(mons[i]->us_op.get());
        if(stats) {
            pva::NetStats::Stats cur;
            stats->stats(cur);

            if(cur.populated) {
                ent.transportPeer = cur.transportPeer;

                stats_add(ent, cur, mons[i]->prevStats, period);
                mons[i]->prevStats = cur;
                us.push_back(ent);
            }
        }

        GWMon::Requester::strong_t dsmons;
        mons[i]->latch(dsmons);

        for(size_t s=0; s<dsmons.size(); s++)
        {
            const pva::NetStats* stats = dynamic_cast<const pva::NetStats*>(dsmons[s]->getRequester().get());
            if(!stats) continue;

            pva::NetStats::Stats cur;
            stats->stats(cur);
            if(!cur.populated) continue;

            ent.dsname = dsmons[s]->name;

            pva::PeerInfo::const_shared_pointer info;
            pva::ChannelRequester::shared_pointer chreq(dsmons[s]->channel->ds_requester.lock());

            if(chreq)
                info = chreq->getPeerInfo();
            if(info)
                ent.transportAccount = info->account;

            ent.transportPeer = cur.transportPeer;
            stats_add(ent, cur, dsmons[s]->prevStats, period);
            dsmons[s]->prevStats = cur;
            ds.push_back(ent);
        }
    }
}

void GWProvider::prepare()
{
    epics::registerRefCounter("GWProvider", &GWProvider::num_instances);
    epics::registerRefCounter("GWChan", &GWChan::num_instances);
    epics::registerRefCounter("GWChan::Requester", &GWChan::Requester::num_instances);
    epics::registerRefCounter("GWMon", &GWMon::num_instances);
    epics::registerRefCounter("GWMon::Requester", &GWMon::Requester::num_instances);
    epics::registerRefCounter("ProxyPut", &ProxyPut::num_instances);
    epics::registerRefCounter("ProxyPut::Requester", &ProxyPut::Requester::num_instances);
    epics::registerRefCounter("ProxyRPC", &ProxyRPC::num_instances);
    epics::registerRefCounter("ProxyRPC::Requester", &ProxyRPC::Requester::num_instances);
}

void GWProvider::runAudit()
{
    audit_log_t audit;
    Guard G(mutex);
    while(audit_run) {
        audit_log.swap(audit); // move any log entries

        {
            UnGuard U(G);

            // deliver to python
            GWProvider_audit(this, audit);
            audit.clear(); // free while unlocked

            // rate limit audit messages by only delivering a batch every 100ms.
            audit_holdoff.wait(0.1);
            // now wait for the audit queue to be non-empty
            audit_wakeup.wait();
        }
        // locked again
    }
}

#ifdef TRACING
std::ostream& show_time(std::ostream& strm)
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    time_t sec = now.tv_sec;
    char buf[40];
    strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&sec));
    size_t end = strlen(buf);
    PyOS_snprintf(buf+end, sizeof(buf)-end, ".%03u ", unsigned(now.tv_nsec/1000000u));
    buf[sizeof(buf)-1] = '\0';
    strm<<buf;
    return strm;
}
#endif // TRACING
