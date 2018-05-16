#include "p4p_client.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

void PutOp::Req::channelPutConnect(
    const pvd::Status& status,
    pva::ChannelPut::shared_pointer const & channelPut,
    pvd::Structure::const_shared_pointer const & structure)
{
    PutOp::shared_pointer op(owner.lock());
    if(!op)
        return;

    TRACE("put start "<<channelPut->getChannel()->getChannelName()<<" "<<status);

    if(!status.isSuccess()) {
        PyLock L;
        PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", status.getMessage().c_str()));
        op->call_cb(err.get());

    } else {
        /* it sucks, but the only way to handle NTEnum is to have the "meta-data"
         * needed to convert string to index.  And the only way to get that is
         * to fetch the current value beforehand.
         *
         * So the choice is either to always fetch the current value,
         * or only sometimes.  As "sometimes" complicates the python
         * side, we go with "always" as simpler, but less efficient.
         */
        channelPut->get();
    }
}

void PutOp::Req::getDone(
    const pvd::Status& status,
    pva::ChannelPut::shared_pointer const & channelPut,
    pvd::PVStructure::shared_pointer const & pvStructure,
    pvd::BitSet::shared_pointer const & bitSet)
{
    PutOp::shared_pointer op(owner.lock());
    if(!op)
        return;

    TRACE("put have current "<<channelPut->getChannel()->getChannelName()<<" "<<status);

    if(!status.isSuccess()) {
        PyLock L;
        PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", status.getMessage().c_str()));
        op->call_cb(err.get());
        return;
    }

    pvd::PVStructure::shared_pointer val;
    pvd::BitSet::shared_pointer mask(new pvd::BitSet);

    {
        PyLock L;
        try {
            PyRef temp;
            temp.swap(op->pyvalue);
            if(!temp) {
                TRACE("no value!?!?!");
                return;
            }

            if(!PyObject_IsInstance(temp.get(), (PyObject*)P4PValue_type)) {
                // assume callable
                pvd::PVStructurePtr curval(pvd::getPVDataCreate()->createPVStructure(pvStructure));
                curval->copyUnchecked(*pvStructure);
                PyRef pcurval(P4PValue_wrap(P4PValue_type, curval, bitSet));
                PyRef val(PyObject_CallFunctionObjArgs(temp.get(), pcurval.get(), NULL));
                temp.swap(val);
            }
            if(!PyObject_IsInstance(temp.get(), (PyObject*)P4PValue_type)) {
                std::ostringstream msg;
                msg<<"Can't put type \""<<Py_TYPE(temp.get())->tp_name<<"\", only Value";
                PyRef err(PyObject_CallFunction(PyExc_ValueError, "s", msg.str().c_str()));
                op->call_cb(err.get());
                return;
            }
            val = P4PValue_unwrap(temp.get(), mask.get());
            if(val->getStructure()!=pvStructure->getStructure()) {
                // callback returned different Structure.
                //TODO: attempt safe/partial copy into server provided Structure?
                PyRef err(PyObject_CallFunction(PyExc_NotImplementedError, "s", "channelPutConnect() safe copy unimplemneted"));
                op->call_cb(err.get());
                return;
            }
        }catch(std::exception& e) {
            PyErr_Print();
            PyErr_Clear();
            std::cerr<<"Error in channelPutConnect value builder: "<<e.what()<<"\n";
            return;
        }
    }
    assert(!!val);
    mask->set(0);
    TRACE("send "<<channelPut->getChannel()->getChannelName()<<" mask="<<*mask<<" value="<<val);
    channelPut->lastRequest();
    // may call putDone() recursively
    channelPut->put(val, mask);
}


void PutOp::Req::putDone(
    const pvd::Status& status,
    pva::ChannelPut::shared_pointer const & channelPut)
{
    PutOp::shared_pointer op(owner.lock());
    if(!op)
        return;

    PyLock L;
    TRACE("status="<<status);
    if(!op->cb) return;
    PyRef V;

    if(status.isSuccess()) {
        V.reset(Py_None, borrow());
    } else {
        // build Exception instance
        // TODO: create RemoteError type
        V.reset(PyObject_CallFunction(PyExc_RuntimeError, (char*)"s", status.getMessage().c_str()));
    }

    if(!V) {
        PyErr_Print();
        PyErr_Clear();
        std::cerr<<"Error in putDone\n";
    } else {
        op->call_cb(V.get());
    }
}

void PutOp::Req::channelDisconnect(bool destroy)
{
    PutOp::shared_pointer op(owner.lock());
    if(!op)
        return;
    PyLock L;
    TRACE("destroy="<<destroy);
    PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", "Disconnected"));
    op->call_cb(err.get());

    op->cancel();
}

void PutOp::cancel()
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
