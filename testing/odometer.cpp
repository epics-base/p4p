/* Tool for testing get throttling
 *
 * A server which only handles GET.
 * Each GET is answered from a global counter counter.
 */

#include <iostream>

#include <pvxs/server.h>
#include <pvxs/source.h>
#include <pvxs/log.h>

using namespace pvxs;

namespace {

const Value prototype(TypeDef(TypeCode::Struct, {
                                  members::UInt64("value")
                              }).create());

struct OdometerSource : public server::Source {
    std::string name;
    uint64_t counter = 0u;

    explicit OdometerSource(const std::string& name) :name(name) {}
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

                // send back current count
                gop->reply(prototype.cloneEmpty()
                                    .update("value", counter++));
            });

            // must provide data type for GET/PUT op
            op->connect(prototype);
        });
    }
};

} // namespace

int main(int argc, char *argv[])
{
    if(argc<2) {
        std::cerr<<"Usage: "<<argv[0]<<" <pvname>"<<std::endl;
        return 1;
    }

    logger_config_env(); // read $PVXS_LOG

    // run server with only our Source
    server::Config::fromEnv()
            .build()
            .addSource("odommeter", std::make_shared<OdometerSource>(argv[1]))
            .run(); // wait for SIGINT

    return 0;
}
