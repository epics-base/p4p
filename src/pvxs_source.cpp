#include <sstream>
#include <map>

#include <epicsTime.h>

#include "p4p.h"
#include <_p4p.h>

#include <pvxs/log.h>

namespace p4p {

namespace {

DEFINE_LOGGER(_log, "p4p.server.source");

// max entries to track in DynamicSource negative result cache
constexpr size_t maxCache = 1024u;
// expiration time for negative result cache entries
constexpr double expireIn = 10.0; // sec

struct DynamicSource : public server::Source
{
    PyObject* handler;

    mutable epicsMutex negCacheLock;
    std::map<std::string, epicsTime> negCache;

    explicit DynamicSource(PyObject* handler)
        :handler(handler)
    {}

    // Source interface
public:
    virtual void onSearch(Search &op) override final
    {
        epicsTime now(epicsTime::getCurrent());

        Guard G(negCacheLock);

        for(auto& chan : op) {
            // test neg cache
            {
                auto it(negCache.find(chan.name()));
                if(it!=negCache.end()) {
                    if(it->second < now) {
                        // stale
                        negCache.erase(it);
                        it = negCache.end();
                        log_debug_printf(_log, "%p neg miss for %s\n", this, chan.name());
                    } else {
                        log_debug_printf(_log, "%p neg hit for %s\n", this, chan.name());
                        continue;
                    }
                }
            }

            {
                UnGuard U(G);
                PyLock L;

                if(!handler)
                    break;

                auto ret(PyRef::allownull(PyObject_CallMethod(handler, "testChannel", "s", chan.name())));
                if(!ret.obj) {
                    PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                    PyErr_Print();
                    PyErr_Clear();

                } else if(PyObject_IsTrue(ret.obj)) {
                    chan.claim();
                    continue;

                } else if(PyBytes_Check(ret.obj) && strcmp(PyBytes_AsString(ret.obj), "nocache")==0) {
                    continue;
                }
            }

            // add to neg cache
            negCache[chan.name()] = now + expireIn;
        }
    }
    virtual void onCreate(std::unique_ptr<server::ChannelControl> &&op) override final
    {
        PyLock L;

        if(!handler)
            return;

        auto ret(PyRef::allownull(PyObject_CallMethod(handler, "makeChannel", "ss",
                                                      op->name().c_str(),
                                                      op->peerName().c_str())));
        if(!ret.obj) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();

        } else if(auto pv = SharedPV_unwrap(ret.obj)) {
            pv.attach(std::move(op));

        } else {
            PyErr_Format(PyExc_TypeError, "makeChannel(\"%s\") must return SharedPV, not %s",
                         op->name().c_str(),
                         Py_TYPE(ret.obj)->tp_name);
            PyErr_Print();
            PyErr_Clear();
        }
    }
};

} // namespace

std::shared_ptr<server::Source> createDynamic(PyObject* handler)
{
    return std::make_shared<DynamicSource>(handler);
}

void disconnectDynamic(const std::shared_ptr<server::Source>& src)
{
    if(!src)
        return;

    auto dynsrc = dynamic_cast<DynamicSource*>(src.get());

    dynsrc->handler = nullptr;
}

}
