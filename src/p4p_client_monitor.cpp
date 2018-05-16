
#include "p4p_client.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

void MonitorOp::Req::monitorConnect(pvd::Status const & status,
                                    pvd::MonitorPtr const & monitor,
                                    pvd::StructureConstPtr const & structure)
{
    TRACE("status="<<status);
    MonitorOp::shared_pointer op(owner);
    if(!op)
        return;
    PyLock L;
    if(op->done) {
        TRACE("completed monitor re-connects?!?");
        return;
    }

    if(status.isSuccess()) {
        monitor->start();
        TRACE("start() "<<op->event.get());
    }

    if(!status.isSuccess()) {
        PyRef err(PyObject_CallFunction(PyExc_RuntimeError, "s", status.getMessage().c_str()));
        op->call_cb(err.get());
        op->event.reset();
        TRACE("error");
    } else {
        op->empty = true;
    }
}

void MonitorOp::Req::monitorEvent(pvd::MonitorPtr const & monitor)
{
    TRACE("");
    MonitorOp::shared_pointer op(owner);
    if(!op)
        return;
    PyLock L;
    op->empty = false;
    PyRef val(Py_True, borrow());
    op->call_cb(val.get());
    TRACE("notified");
}

void MonitorOp::Req::unlisten(epics::pvAccess::Monitor::shared_pointer const & monitor)
{
    TRACE("");
    MonitorOp::shared_pointer op(owner);
    if(!op)
        return;
    PyLock L;
    op->done = true;
    PyRef val(Py_False, borrow());
    op->call_cb(val.get());
    (void)op->op->stop();
    op->op->destroy();
    op->op.reset();
}

void MonitorOp::Req::channelDisconnect(bool destroy)
{
    TRACE("");
    MonitorOp::shared_pointer op(owner);
    if(!op)
        return;
    PyLock L;

    PyRef val(Py_None, borrow());
    op->call_cb(val.get());
}

void MonitorOp::call_cb(PyObject *obj) {
    if(!event.get()) return;
    PyObject *junk = PyObject_CallFunctionObjArgs(event.get(), obj, NULL);
    if(junk) {
        Py_DECREF(junk);
    } else {
        PyErr_Print();
        PyErr_Clear();
    }
}

MonitorOp::MonitorOp()
    :op()
    ,event()
    ,empty(true)
    ,done(false)
{}

MonitorOp::~MonitorOp()
{
    // TODO: notify event?
    operation_t::shared_pointer O;
    op.swap(O);
    if(O) {
        PyUnlock U;
        O->destroy();
        O.reset();
    }
}

namespace {

#define TRY PyMonitorOp::reference_type SELF = PyMonitorOp::unwrap(self); try

PyObject *py_monitor_close(PyObject *self)
{
    TRY {
        TRACE("cancel subscription");
        SELF->event.reset();
        pva::Monitor::shared_pointer op;
        SELF->op.swap(op);
        if(op) {
            PyUnlock U;
            op->stop();
            op->destroy();
            op.reset();
        }

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject *py_monitor_empty(PyObject *self)
{
    TRY {
        TRACE(SELF->empty);
        if(SELF->empty)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;

    }CATCH()
    return NULL;
}

PyObject *py_monitor_done(PyObject *self)
{
    TRY {
        TRACE(SELF->done);
        if(SELF->done)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;

    }CATCH()
    return NULL;
}

PyObject *py_monitor_pop(PyObject *self)
{
    TRY {
        if(!SELF->op)
            Py_RETURN_NONE;

        pva::MonitorElementPtr elem(SELF->op->poll());
        SELF->empty = !elem;
        if(!elem) {
            TRACE("Empty");
            Py_RETURN_NONE;
        }
        try {

            const pvd::PVStructure::shared_pointer& E = elem->pvStructurePtr;

            pvd::PVStructure::shared_pointer V(pvd::getPVDataCreate()->createPVStructure(E->getStructure()));
            V->copyUnchecked(*E);

            pvd::BitSet::shared_pointer M;
            if(elem->changedBitSet) {
                M.reset(new pvd::BitSet);
                *M = *elem->changedBitSet;
            }
            // TODO: overrunBitSet??

            SELF->op->release(elem);

            TRACE("event="<<V);
            return P4PValue_wrap(P4PValue_type, V, M);
        } catch(...){
            SELF->op->release(elem);
            throw;
        }

    }CATCH()
    return NULL;
}

int py_monitor_traverse(PyObject *self, visitproc visit, void *arg)
{
    TRY {
        if(SELF->event)
            Py_VISIT(SELF->event.get());
        return 0;
    }CATCH()
    return -1;
}
int py_monitor_clear(PyObject *self)
{
    TRY {
        TRACE("sub clear");
        PyRef tmp;
        SELF->event.swap(tmp);
        return 0;
    }CATCH()
    return -1;
}

static PyMethodDef PyMonitorOp_methods[] = {
    {"close", (PyCFunction)&py_monitor_close, METH_NOARGS,
     "Cancel subscription."},
    {"empty", (PyCFunction)&py_monitor_empty, METH_NOARGS,
     "Would pop() return a value?"},
    {"done", (PyCFunction)&py_monitor_done, METH_NOARGS,
     "Has the last subscription update been received?  Check after pop() returns None."},
    {"pop", (PyCFunction)&py_monitor_pop, METH_NOARGS,
     "Pull an entry from the subscription queue.  return None if empty"},
    {NULL}
};

} // namespace

template<>
PyTypeObject PyMonitorOp::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Subscription",
    sizeof(PyMonitorOp),
};

void p4p_client_monitor_register(PyObject *mod)
{
    PyMonitorOp::buildType();
    PyMonitorOp::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    PyMonitorOp::type.tp_traverse = &py_monitor_traverse;
    PyMonitorOp::type.tp_clear = &py_monitor_clear;

    PyMonitorOp::type.tp_methods = PyMonitorOp_methods;

    if(PyType_Ready(&PyMonitorOp::type))
        throw std::runtime_error("failed to initialize PyMonitorOp");

    Py_INCREF((PyObject*)&PyMonitorOp::type);
    if(PyModule_AddObject(mod, "Subscription", (PyObject*)&PyMonitorOp::type)) {
        Py_DECREF((PyObject*)&PyOp::type);
        throw std::runtime_error("failed to add p4p._p4p.Subscription");
    }
}
