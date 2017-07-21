#include "p4p_client.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

void RPCOp::Req::channelRPCConnect(
    const pvd::Status& status,
    pva::ChannelRPC::shared_pointer const & channelRPC)
{
    RPCOp::shared_pointer op(owner.lock());
    if(!op)
        return;

    TRACE("rpc start "<<channelRPC->getChannel()->getChannelName()<<" "<<status);

    if(!status.isSuccess()) {
        PyLock L;
        PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", status.getMessage().c_str()));
        op->call_cb(err.get());
    } else {
        pvd::PVStructure::shared_pointer val;
        {
            PyLock L;
            op->pvvalue.swap(val);
        }

        if(val) {
            channelRPC->lastRequest();
            channelRPC->request(val);
        }
    }
}

void RPCOp::Req::requestDone(
    const pvd::Status& status,
    pva::ChannelRPC::shared_pointer const & channelRPC,
    pvd::PVStructure::shared_pointer const & pvResponse)
{
    RPCOp::shared_pointer op(owner.lock());
    if(!op)
        return;

    TRACE("rpc done "<<channelRPC->getChannel()->getChannelName()<<" "<<status);

    PyLock L;

    if(!status.isSuccess()) {
        PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", status.getMessage().c_str()));
        op->call_cb(err.get());
    } else {
        PyRef val(P4PValue_wrap(P4PValue_type, pvResponse));
        op->call_cb(val.get());
    }
}

void RPCOp::Req::channelDisconnect(bool destroy)
{
    RPCOp::shared_pointer op(owner.lock());
    if(!op)
        return;
    PyLock L;
    TRACE("destroy="<<destroy);
    PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", "Disconnected"));
    op->call_cb(err.get());

    op->cancel();
}

void RPCOp::cancel()
{
    operation_t::shared_pointer O;
    op.swap(O);
    if(O) {
        PyUnlock U;
        O->destroy();
        O.reset();
    }
    OpBase::cancel();
}
