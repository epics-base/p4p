#include "p4p_client.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

void GetOp::Req::channelGetConnect(
    const pvd::Status& status,
    pva::ChannelGet::shared_pointer const & channelGet,
    pvd::Structure::const_shared_pointer const & structure)
{
    GetOp::shared_pointer op(owner.lock());
    TRACE("get start "<<channelGet->getChannel()->getChannelName()<<" "<<status<<" "<<op);
    if(!op)
        return;

    if(!status.isSuccess()) {
        PyLock L;
        PyRef E(PyObject_CallFunction(PyExc_RuntimeError, (char*)"s", status.getMessage().c_str()));
        op->call_cb(E.get());
    } else {
        channelGet->lastRequest();
        // may call getDone() recursively
        channelGet->get();
    }
}

void GetOp::Req::getDone(
    const pvd::Status& status,
    pva::ChannelGet::shared_pointer const & channelGet,
    pvd::PVStructure::shared_pointer const & pvStructure,
    pvd::BitSet::shared_pointer const & bitSet)
{
    GetOp::shared_pointer op(owner.lock());
    if(!op)
        return;
    PyLock L;

    TRACE("get complete "<<channelGet->getChannel()->getChannelName()<<" for "<<op->cb.get()<<" with "<<status);
    if(!op->cb.get()) return;
    PyRef V;

    if(status.isSuccess()) {
        // we don't re-use ChannelGet, so assume exclusive ownership of pvStructure w/o a copy
        V.reset(P4PValue_wrap(P4PValue_type, pvStructure));
    } else {
        // build Exception instance
        // TODO: create RemoteError type
        V.reset(PyObject_CallFunction(PyExc_RuntimeError, (char*)"s", status.getMessage().c_str()));
    }

    if(!V.get()) {
        PyErr_Print();
        PyErr_Clear();
    } else {
        op->call_cb(V.get());
    }
}

void GetOp::Req::channelDisconnect(bool destroy)
{
    GetOp::shared_pointer op(owner.lock());
    if(!op)
        return;
    PyLock L;
    TRACE("destroy="<<destroy);
    PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", "Disconnected"));
    op->call_cb(err.get());

    op->cancel();
}

void GetOp::cancel()
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
