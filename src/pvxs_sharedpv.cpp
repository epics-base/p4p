
#include <algorithm>
#include <mutex>
#include <vector>
#include <sstream>

#include "p4p.h"
#include <_p4p.h>

namespace p4p {

std::string toString(const server::Server& serv, int detail)
{
    std::ostringstream strm;
    Detailed D(strm, detail);
    strm<<serv;
    return strm.str();
}

void attachHandler(server::SharedPV& pv, PyObject *handler)
{
    // we assume the caller will maintain a ref. to handler until detachHandler() returns

    {
        PyUnlock U;

        pv.onFirstConnect([handler](server::SharedPV&){
            PyLock L;

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "onFirstConnect", "")));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
                // continue to build the channel
            }
        });

        pv.onLastDisconnect([handler](server::SharedPV&){
            PyLock L;

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "onLastDisconnect", "")));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
                // continue to build the channel
            }
        });

        pv.onPut([handler](server::SharedPV&, std::unique_ptr<server::ExecOp>&& rawop, Value&& val){
            PyLock L;
            std::shared_ptr<server::ExecOp> op(std::move(rawop));
            PyRef pyop(ServerOperation_wrap(op, val));

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "put", "O", pyop.obj)));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                op->error("Internal Error on Remote end");
            }
        });

        pv.onRPC([handler](server::SharedPV&, std::unique_ptr<server::ExecOp>&& rawop, Value&& val){
            PyLock L;
            std::shared_ptr<server::ExecOp> op(std::move(rawop));
            PyRef pyop(ServerOperation_wrap(op, val));

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "rpc", "O", pyop.obj)));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                op->error("Internal Error on Remote end");
            }
        });
    }
}

void detachHandler(server::SharedPV& pv)
{
    PyUnlock U;

    pv.onFirstConnect(nullptr);

    pv.onLastDisconnect(nullptr);

    pv.onPut(nullptr);

    pv.onRPC(nullptr);
}

void attachCleanup(const std::shared_ptr<server::ExecOp>& op, PyObject *handler)
{
    PyUnlock U;

    op->onCancel([handler]() {
        PyLock L;

        auto ret(PyRef::allownull(PyObject_CallFunction(handler, "")));
        if(PyErr_Occurred()) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();
        }
    });
}

void detachCleanup(const std::shared_ptr<server::ExecOp> &op)
{
    PyUnlock U;
    op->onCancel(nullptr);
}

struct GetInterceptSource : public server::Source,
                             public std::enable_shared_from_this<GetInterceptSource> {
    std::string pvname;
    server::SharedPV pv;
    // handler is borrowed — caller (Cython StaticProvider) maintains lifetime
    PyObject* handler;

    // Active channels, kept alive here so PVXS can dispatch callbacks.
    // Removed via onClose when a client disconnects; all remaining are
    // cleared when this source is destroyed (after server.stop()).
    std::mutex channels_lock;
    std::vector<std::shared_ptr<server::ChannelControl>> channels;

    GetInterceptSource(const std::string& name_, server::SharedPV& pv_, PyObject* handler_)
        : pvname(name_), pv(pv_), handler(handler_)
    {}

    virtual ~GetInterceptSource() {}

    virtual void onSearch(server::Source::Search& op) override {
        for(auto& name : op) {
            if(name.name() == pvname) {
                PyLock L;
                if(pv.isOpen())
                    name.claim();
            }
        }
    }

    virtual void onCreate(std::unique_ptr<server::ChannelControl>&& op) override {
        auto ctrl = std::shared_ptr<server::ChannelControl>(std::move(op));

        // Use weak_ptr so the onClose callback doesn't extend our lifetime,
        // and so it's a no-op if we're already being destroyed.
        std::weak_ptr<GetInterceptSource> weak_self(shared_from_this());
        server::ChannelControl* raw_ctrl = ctrl.get();

        ctrl->onClose([weak_self, raw_ctrl](const std::string&) {
            if(auto self = weak_self.lock()) {
                std::lock_guard<std::mutex> lock(self->channels_lock);
                auto& ch = self->channels;
                ch.erase(
                    std::remove_if(ch.begin(), ch.end(),
                        [raw_ctrl](const std::shared_ptr<server::ChannelControl>& c) {
                            return c.get() == raw_ctrl;
                        }),
                    ch.end()
                );
            }
        });

        ctrl->onOp([this](std::unique_ptr<server::ConnectOp>&& connectop) {
            auto cop = std::shared_ptr<server::ConnectOp>(std::move(connectop));

            // Tell client what data type this PV provides
            {
                PyLock L;
                if(!pv.isOpen()) {
                    cop->error("PV not open");
                    return;
                }
                Value proto;
                {
                    PyUnlock U;
                    proto = pv.fetch();
                }
                cop->connect(proto);
            }

            // GET handler
            cop->onGet([this](std::unique_ptr<server::ExecOp>&& gop) {
                PyLock L;
                std::shared_ptr<server::ExecOp> op(std::move(gop));

                if(!PyObject_HasAttrString(handler, "onGet")) {
                    // CPP-03: no onGet — return current PV value (backward compat)
                    Value current;
                    {
                        PyUnlock U;
                        current = pv.fetch();
                        op->reply(current);
                    }
                    return;
                }

                PyRef pyop(ServerOperation_wrap(op, Value{}));

                auto ret(PyRef::allownull(PyObject_CallMethod(handler, "onGet", "O", pyop.obj)));
                if(PyErr_Occurred()) {
                    PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                    PyErr_Print();
                    PyErr_Clear();
                    op->error("Internal Error on Remote end");
                }
            });

            // PUT handler — delegate to Python handler.put
            cop->onPut([this](std::unique_ptr<server::ExecOp>&& rawop, Value&& val) {
                PyLock L;
                std::shared_ptr<server::ExecOp> op(std::move(rawop));
                PyRef pyop(ServerOperation_wrap(op, val));

                auto ret(PyRef::allownull(PyObject_CallMethod(handler, "put", "O", pyop.obj)));
                if(PyErr_Occurred()) {
                    PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                    PyErr_Print();
                    PyErr_Clear();
                    op->error("Internal Error on Remote end");
                }
            });
        });

        // RPC handler — delegate to Python handler.rpc
        ctrl->onRPC([this](std::unique_ptr<server::ExecOp>&& rawop, Value&& val) {
            PyLock L;
            std::shared_ptr<server::ExecOp> op(std::move(rawop));
            PyRef pyop(ServerOperation_wrap(op, val));

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "rpc", "O", pyop.obj)));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                op->error("Internal Error on Remote end");
            }
        });

        // Subscribe handler — deliver current value on connect
        ctrl->onSubscribe([this](std::unique_ptr<server::MonitorSetupOp>&& mop) {
            PyLock L;
            if(!pv.isOpen()) {
                mop->error("PV not open");
                return;
            }
            Value proto;
            {
                PyUnlock U;
                proto = pv.fetch();
            }
            auto sub = mop->connect(proto);
            sub->post(proto);
        });

        // Store ctrl to keep the ChannelControl alive while the channel is open.
        // This avoids a self-referential shared_ptr cycle (ctrl captured in its own
        // lambda) that prevents cleanup on some platforms.  Removed via onClose above,
        // or flushed when this source is destroyed.
        std::lock_guard<std::mutex> lock(channels_lock);
        channels.push_back(std::move(ctrl));
    }
};

std::shared_ptr<server::Source> attachGetHandler(const std::string& name, server::SharedPV& pv, PyObject* handler)
{
    return std::make_shared<GetInterceptSource>(name, pv, handler);
}

}
