/* Tool for testing get throttling
 *
 * A server which only handles GET.
 * Each GET is answered from a global counter counter.
 */

#if !defined(_WIN32)
#include <signal.h>
#define USE_SIGNAL
#endif

#include <iostream>

#include <epicsGuard.h>
#include <epicsEvent.h>

#include <pv/sharedPtr.h>
#include <pv/pvAccess.h>
#include <pv/serverContext.h>
#include <pv/logger.h>

namespace  {

epicsEvent done;

#ifdef USE_SIGNAL
void alldone(int num)
{
    (void)num;
    done.signal();
}
#endif

typedef epicsGuard<epicsMutex> Guard;

namespace pva = epics::pvAccess;
namespace pvd = epics::pvData;

struct CounterProvider;
struct CounterChannel;

pvd::StructureConstPtr counterType(pvd::FieldBuilder::begin()
                                   ->add("value", pvd::pvULong)
                                   ->createStructure());

struct CounterGet : public pva::ChannelGet,
                    public std::tr1::enable_shared_from_this<CounterGet>
{
    const std::tr1::shared_ptr<CounterChannel> channel;
    const pva::ChannelGetRequester::weak_pointer requester;

    pvd::PVStructure::shared_pointer inst;
    pvd::BitSet::shared_pointer changed;

    CounterGet(const std::tr1::shared_ptr<CounterChannel>& channel,
               const pva::ChannelGetRequester::shared_pointer& requester)
        :channel(channel)
        ,requester(requester)
        ,inst(counterType->build())
        ,changed(new pvd::BitSet(inst->getNumberFields()))
    {}
    virtual ~CounterGet() {}

    virtual void destroy() OVERRIDE FINAL {}

    virtual pva::Channel::shared_pointer getChannel() OVERRIDE FINAL;
    virtual void cancel() OVERRIDE FINAL {}
    virtual void lastRequest() OVERRIDE FINAL {}

    virtual void get() OVERRIDE FINAL;
};

struct CounterChannel : public pva::Channel,
                        public std::tr1::enable_shared_from_this<CounterChannel>
{
    const std::tr1::weak_ptr<CounterProvider> provider;
    const pva::ChannelRequester::weak_pointer requester;

    CounterChannel(const std::tr1::shared_ptr<CounterProvider>& provider,
                   const pva::ChannelRequester::shared_pointer& requester)
        :provider(provider)
        ,requester(requester)
    {}
    virtual ~CounterChannel() {}

    virtual std::string getRequesterName() OVERRIDE FINAL { return "odometer"; }

    virtual void destroy() OVERRIDE FINAL {}

    virtual pva::ChannelProvider::shared_pointer getProvider() OVERRIDE FINAL;
    virtual std::string getRemoteAddress() OVERRIDE FINAL { return "something"; }
    virtual std::string getChannelName() OVERRIDE FINAL;
    virtual pva::ChannelRequester::shared_pointer getChannelRequester() OVERRIDE FINAL
    { return pva::ChannelRequester::shared_pointer(requester); }

    virtual pva::ChannelGet::shared_pointer createChannelGet(
            pva::ChannelGetRequester::shared_pointer const & requester,
            pvd::PVStructure::shared_pointer const & pvRequest) OVERRIDE FINAL
    {
        std::tr1::shared_ptr<CounterGet> ret(new CounterGet(shared_from_this(), requester));
        requester->channelGetConnect(pvd::Status(), ret, counterType);
        return ret;
    }
};

struct CounterProvider : public pva::ChannelProvider,
                         public std::tr1::enable_shared_from_this<CounterProvider>
{
    const std::string pvname;

    mutable epicsMutex mutex;
    pvd::uint64 counter;

    explicit CounterProvider(const std::string& pvname)
        :pvname(pvname)
        ,counter(0u)
    {}
    virtual ~CounterProvider() {}

    virtual void destroy() OVERRIDE FINAL {}
    virtual std::string getProviderName() OVERRIDE FINAL { return "odometer"; }

    virtual pva::ChannelFind::shared_pointer channelFind(std::string const & name,
                                                         pva::ChannelFindRequester::shared_pointer const & requester) OVERRIDE FINAL
    {
        pva::ChannelFind::shared_pointer ret(pva::ChannelFind::buildDummy(shared_from_this()));
        requester->channelFindResult(pvd::Status(), ret, name==pvname);
        return ret;
    }

    virtual pva::Channel::shared_pointer createChannel(std::string const & name,
                                                       pva::ChannelRequester::shared_pointer const & requester,
                                                       short priority, std::string const & address) OVERRIDE FINAL
    {
        pva::Channel::shared_pointer ret;
        if(name!=pvname) {
            requester->channelCreated(pvd::Status::error("Not such PV"), ret);

        } else {
            std::tr1::shared_ptr<CounterChannel> chan(new CounterChannel(shared_from_this(), requester));
            ret = chan;
            requester->channelCreated(pvd::Status(), ret);
        }
        return ret;
    }
};

void CounterGet::get()
{
    std::tr1::shared_ptr<pva::ChannelGetRequester> req(requester);
    std::tr1::shared_ptr<CounterProvider> prov(channel->provider);

    if(!req || !prov)
        return;

    pvd::uint64 cnt;
    {
        Guard G(prov->mutex);
        cnt = prov->counter++;
    }

    changed->clear();

    pvd::PVULong::shared_pointer fld = inst->getSubFieldT<pvd::PVULong>("value");
    fld->put(cnt);
    changed->set(fld->getFieldOffset());

    req->getDone(pvd::Status(), shared_from_this(), inst, changed);
}

pva::Channel::shared_pointer CounterGet::getChannel()
{
    return pva::Channel::shared_pointer(channel);
}

pva::ChannelProvider::shared_pointer CounterChannel::getProvider()
{ return std::tr1::shared_ptr<CounterProvider>(provider); }

std::string CounterChannel::getChannelName()
{
    std::tr1::shared_ptr<CounterProvider> prov(provider);
    if(prov)
        return prov->pvname;
    else
        return "";
}

} // namespace


int main(int argc, char *argv[])
{
    if(argc<2) {
        std::cerr<<"Usage: "<<argv[0]<<" <pvname>"<<std::endl;
        return 1;
    }

    SET_LOG_LEVEL(pva::logLevelAll);

    std::tr1::shared_ptr<CounterProvider> provider(new CounterProvider(argv[1]));

    pva::ServerContext::shared_pointer server(pva::ServerContext::create(pva::ServerContext::Config()
                                                                         .provider(provider)
                                                                         ));

#ifdef USE_SIGNAL
        signal(SIGINT, alldone);
        signal(SIGTERM, alldone);
        signal(SIGQUIT, alldone);
#endif

    done.wait();

    return 0;
}
