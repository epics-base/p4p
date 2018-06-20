

#include "p4p.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

typedef PyClassWrapper<pvas::Operation> PyOperation;
typedef PyClassWrapper<pvas::SharedPV::shared_pointer> PySharedPV;

PyClassWrapper_DEF(PyOperation, "ServerOperation")
PyClassWrapper_DEF(PySharedPV, "SharedPV")

namespace {

struct PVHandler : public pvas::SharedPV::Handler {
    POINTER_DEFINITIONS(PVHandler);

    static size_t num_instances;

    PyRef cb;

    PVHandler(PyObject *cb) : cb(cb, borrow()) {
        REFTRACE_INCREMENT(num_instances);
        TRACE("");
    }
    virtual ~PVHandler() {
        // we may get here with the GIL locked (via ~PySharedPV), or not (SharedPV released from ServerContext)
        PyLock L;
        TRACE("");
        cb.reset();
        REFTRACE_DECREMENT(num_instances);
    }

    virtual void onFirstConnect(const pvas::SharedPV::shared_pointer& pv) OVERRIDE FINAL {
        PyLock L;
        TRACE(cb.get());
        if(!cb) return;

        PyRef ret(PyObject_CallMethod(cb.get(), "onFirstConnect", ""), allownull());
        if(PyErr_Occurred()) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
            // we don't treat this failure as critical.
            // continue to build the channel
        }
    }

    virtual void onLastDisconnect(const pvas::SharedPV::shared_pointer& pv) OVERRIDE FINAL {
        PyLock L;
        TRACE(cb.get());
        if(!cb) return;

        PyRef ret(PyObject_CallMethod(cb.get(), "onLastDisconnect", ""), allownull());
        if(PyErr_Occurred()) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
            // we don't treat this failure as critical.
            // continue to build the channel
        }
    }

    virtual void onPut(const pvas::SharedPV::shared_pointer& pv, pvas::Operation& op) OVERRIDE FINAL {
        {
            PyLock L;
            TRACE(cb.get());
            if(!cb) return;

            PyRef args(PyTuple_New(0));
            PyRef kws(PyDict_New());
            PyRef pyop(PyOperation::type.tp_new(&PyOperation::type, args.get(), kws.get()));

            PyOperation::unwrap(pyop.get()) = op;

            PyRef ret(PyObject_CallMethod(cb.get(), "put", "O", pyop.get()), allownull());
            if(PyErr_Occurred()) {
                TRACE("ERROR");
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
            } else {
                return;
            }
        }
        op.complete(pvd::Status::error("Internal Error on Remote end"));
    }

    virtual void onRPC(const pvas::SharedPV::shared_pointer& pv, pvas::Operation& op) OVERRIDE FINAL {
        {
            PyLock L;
            TRACE("");

            PyRef args(PyTuple_New(0));
            PyRef kws(PyDict_New());
            PyRef pyop(PyOperation::type.tp_new(&PyOperation::type, args.get(), kws.get()));

            PyOperation::unwrap(pyop.get()) = op;

            PyRef ret(PyObject_CallMethod(cb.get(), "rpc", "O", pyop.get()), allownull());
            if(PyErr_Occurred()) {
                TRACE("ERROR");
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
            } else {
                return;
            }
        }
        op.complete(pvd::Status::error("Internal Error on Remote end"));
    }
}; // PVHandler

size_t PVHandler::num_instances;

#define TRY PySharedPV::reference_type SELF = PySharedPV::unwrap(self); try

static int sharedpv_init(PyObject* self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"handler", NULL};
        PyObject *handler = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", (char**)names, &handler))
            return -1;

        if(SELF) {
            // already set by P4PSharedPV_wrap()
        } else if(handler != Py_None) {
            PVHandler::shared_pointer H(new PVHandler(handler));

            SELF = pvas::SharedPV::build(H);
        } else {
            SELF = pvas::SharedPV::buildReadOnly();
        }

        return 0;
    }CATCH()
    return -1;
}

static PyObject* sharedpv_open(PyObject* self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"value", NULL};
        PyObject *value;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!", (char**)names, P4PValue_type, &value))
            return NULL;

        pvd::BitSet changed;
        pvd::PVStructurePtr S(P4PValue_unwrap(value, &changed));

        TRACE("");
        {
            PyUnlock U;
            SELF->open(*S, changed);
        }
        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static PyObject* sharedpv_post(PyObject* self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"value", NULL};
        PyObject *value;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!", (char**)names, P4PValue_type, &value))
            return NULL;

        pvd::BitSet changed;
        pvd::PVStructurePtr S(P4PValue_unwrap(value, &changed));

        TRACE("");
        {
            PyUnlock U;
            SELF->post(*S, changed);
        }
        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static PyObject* sharedpv_close(PyObject* self) {
    TRY {
        TRACE("");
        {
            PyUnlock U;
            SELF->close();
        }
        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static PyObject* sharedpv_isOpen(PyObject* self) {
    TRY {
        TRACE("");
        if(SELF->isOpen())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }CATCH()
    return NULL;
}

static int sharedpv_traverse(PyObject *self, visitproc visit, void *arg)
{
    TRY {
        if(!SELF) return 0; // eg. failed sub-class ctor
        // attempt to drill through to the handler
        PVHandler::shared_pointer handler(std::tr1::dynamic_pointer_cast<PVHandler>(SELF->getHandler()));
        if(handler && handler->cb)
            Py_VISIT(handler->cb.get());
        return 0;
    } CATCH()
    return -1;
}

static int sharedpv_clear(PyObject *self)
{
    TRY {
        if(!SELF) return 0; // eg. failed sub-class ctor
        // also called by PyClassWrapper dtor, so we are killing the Handler
        // even though it may still be ref'd
        PVHandler::shared_pointer handler(std::tr1::dynamic_pointer_cast<PVHandler>(SELF->getHandler()));
        // ~= Py_CLEAR(cb)
        if(handler) {
            PyRef tmp;
            handler->cb.swap(tmp);
        }
        return 0;
    } CATCH()
    return -1;
}

static PyMethodDef SharedPV_methods[] = {
    {"open", (PyCFunction)&sharedpv_open, METH_VARARGS|METH_KEYWORDS,
     "Mark PV as opened and provide initial value"},
    {"post", (PyCFunction)&sharedpv_post, METH_VARARGS|METH_KEYWORDS,
     "Update value of PV"},
    {"close", (PyCFunction)&sharedpv_close, METH_NOARGS,
     "Mark PV as closed"},
    {"isOpen", (PyCFunction)&sharedpv_isOpen, METH_NOARGS,
     "Has open() been called?"},
    {NULL}
};

#undef TRY
#define TRY PyOperation::reference_type SELF = PyOperation::unwrap(self); try

PyObject* operation_pvRequest(PyObject *self)
{
    TRY {
        const pvd::PVStructure& s = SELF.pvRequest();
        pvd::PVStructurePtr S(pvd::getPVDataCreate()->createPVStructure(s.getStructure()));
        S->copyUnchecked(s);
        return P4PValue_wrap(P4PValue_type, S);
    } CATCH()
    return NULL;
}

PyObject* operation_value(PyObject *self)
{
    TRY {
        const pvd::PVStructure& s = SELF.value();
        pvd::PVStructurePtr S(pvd::getPVDataCreate()->createPVStructure(s.getStructure()));
        S->copyUnchecked(s);
        pvd::BitSetPtr C(new pvd::BitSet(SELF.changed()));
        return P4PValue_wrap(P4PValue_type, S, C);
    } CATCH()
    return NULL;
}

PyObject* operation_done(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"value", "error", NULL};
        PyObject *value = Py_None;
        const char *error = NULL;

        if(!PyArg_ParseTupleAndKeywords(args, kws, "|Oz", (char**)names, &value, &error))
            return NULL;

        if(error) {
            TRACE("done with error "<<error);
            PyUnlock U;
            SELF.complete(pvd::Status::error(error));

        } else if(value == Py_None) {
            TRACE("done without data");
            PyUnlock U;
            SELF.complete();

        } else if(PyObject_IsInstance(value, (PyObject*)P4PValue_type)) {
            pvd::BitSet C;
            pvd::PVStructurePtr S(P4PValue_unwrap(value, &C));

            TRACE("done with data");
            PyUnlock U;
            SELF.complete(*S, C);

        } else {
            return PyErr_Format(PyExc_TypeError, "Invalid arguments");
        }

        Py_RETURN_NONE;
    } CATCH()
    return NULL;
}

PyObject* operation_info(PyObject *self, PyObject *args)
{
    TRY {
        const char *msg;

        if(!PyArg_ParseTuple(args, "s", &msg))
            return NULL;

        TRACE("");
        {
            PyUnlock U;
            SELF.info(msg);
        }
        Py_RETURN_NONE;
    } CATCH()
    return NULL;
}

PyObject* operation_warn(PyObject *self, PyObject *args)
{
    TRY {
        const char *msg;

        if(!PyArg_ParseTuple(args, "s", &msg))
            return NULL;

        TRACE("");
        {
            PyUnlock U;
            SELF.warn(msg);
        }
        Py_RETURN_NONE;
    } CATCH()
    return NULL;
}

static PyMethodDef Operation_methods[] = {
    {"pvRequest", (PyCFunction)&operation_pvRequest, METH_NOARGS,
     "Access to client provided request modifiers"},
    {"value", (PyCFunction)&operation_value, METH_NOARGS,
     "Access to client provided input/argument value"},
    {"done", (PyCFunction)&operation_done, METH_VARARGS|METH_KEYWORDS,
     "Complete in-progress operation"},
    {"info", (PyCFunction)&operation_info, METH_VARARGS,
     "Send remote info message"},
    {"warn", (PyCFunction)&operation_warn, METH_VARARGS,
     "Send remote warning message"},
    {NULL}
};

} //namespace

PyTypeObject* P4PSharedPV_type = &PySharedPV::type;

std::tr1::shared_ptr<pvas::SharedPV> P4PSharedPV_unwrap(PyObject *obj)
{
    return PySharedPV::unwrap(obj);
}
PyObject *P4PSharedPV_wrap(const std::tr1::shared_ptr<pvas::SharedPV>& pv)
{
    assert(!!pv);
    PyTypeObject *type = P4PSharedPV_type;
    PyRef args(PyTuple_New(0));
    PyRef kws(PyDict_New());

    PyRef ret(type->tp_new(type, args.get(), kws.get()));

    // inject value *before* __init__ of base or derived type runs
    PySharedPV::unwrap(ret.get()) = pv;

    if(type->tp_init(ret.get(), args.get(), kws.get()))
        throw std::runtime_error("XXX");

    return ret.release();
}

void p4p_server_sharedpv_register(PyObject *mod)
{
    PySharedPV::buildType();
    PySharedPV::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC;
    PySharedPV::type.tp_init = &sharedpv_init;
    PySharedPV::type.tp_traverse = &sharedpv_traverse;
    PySharedPV::type.tp_clear = &sharedpv_clear;

    PySharedPV::type.tp_methods = SharedPV_methods;

    PySharedPV::finishType(mod, "SharedPV");

    PyOperation::buildType();
    PyOperation::type.tp_flags = Py_TPFLAGS_DEFAULT;
    // no GC

    PyOperation::type.tp_methods = Operation_methods;

    PyOperation::finishType(mod, "ServerOperation");

    epics::registerRefCounter("p4p._p4p.SharedPV::Handler", &PVHandler::num_instances);
}
