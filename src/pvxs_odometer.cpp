/* Tool for testing get throttling
 *
 * A server which only handles GET.
 * Each GET is answered from a global counter counter.
 */

#include <iostream>

#include <epicsTime.h>

#include <pvxs/server.h>
#include <pvxs/source.h>
#include <pvxs/nt.h>
#include <pvxs/log.h>

using namespace pvxs;

namespace {

struct OdometerSource : public server::Source {

    std::string name;
    uint64_t counter = 0u;
    const Value prototype;

    explicit OdometerSource(const std::string& name)
        :name(name)
        ,prototype(nt::NTScalar{TypeCode::UInt64}.create())
    {}
    virtual ~OdometerSource() {}

    // client searching
    virtual void onSearch(Search &op) override final
    {
        for(auto& search : op) {
            if(search.name()==name)
                search.claim();
        }
    }

    // client creating a channel
    virtual void onCreate(std::unique_ptr<server::ChannelControl> &&cop) override final
    {
        cop->onOp([this](std::unique_ptr<server::ConnectOp>&& op) { // client starts a GET/PUT operation
            op->onGet([this](std::unique_ptr<server::ExecOp>&& gop) { // execute GET (or Get of PUT)

                epicsTimeStamp now;
                (void)epicsTimeGetCurrent(&now);

                // send back current count
                gop->reply(prototype.cloneEmpty()
                                    .update("value", counter++)
                                    .update("timeStamp.secondsPastEpoch", now.secPastEpoch + POSIX_TIME_AT_EPICS_EPOCH)
                                    .update("timeStamp.nanoseconds", now.nsec));
            });

            // must provide data type for GET/PUT op
            op->connect(prototype);
        });
    }
};

} // namespace

namespace p4p {

std::shared_ptr<server::Source> makeOdometer(const std::string& name) {
    return std::make_shared<OdometerSource>(name);
}

} // namespace p4p
