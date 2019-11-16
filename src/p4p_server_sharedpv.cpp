
#include <pv/security.h>

#include "p4p.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

typedef PyClassWrapper<pvas::Operation, true> PyOperation;
typedef PyClassWrapper<pvas::SharedPV::shared_pointer, true> PySharedPV;

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
            if(!ret) {
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
            TRACE(cb.get());
            if(!cb) return;

            PyRef args(PyTuple_New(0));
            PyRef kws(PyDict_New());
            PyRef pyop(PyOperation::type.tp_new(&PyOperation::type, args.get(), kws.get()));

            PyOperation::unwrap(pyop.get()) = op;

            PyRef ret(PyObject_CallMethod(cb.get(), "rpc", "O", pyop.get()), allownull());
            if(!ret) {
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
        const char *names[] = {"handler", "options", NULL};
        PyObject *handler = Py_None, *pyopts = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", (char**)names, &handler, &pyopts))
            return -1;

        pvas::SharedPV::Config conf;
        if(pyopts!=Py_None) {
            PyRef val;

            // eg. {'dropEmptyUpdates':True}
            val.reset(PyObject_CallMethod(pyopts, "get", "sO", "dropEmptyUpdates", Py_None));
            if(val.get()!=Py_None) {
                conf.dropEmptyUpdates = PyObject_IsTrue(val.get());
            }

            // eg. {'mapperMode':0}
            val.reset(PyObject_CallMethod(pyopts, "get", "sO", "mapperMode", Py_None));
            if(val.get()!=Py_None) {
                PyString pystr(val.get());
                std::string str(pystr.str());
                if(str=="Mask") {
                    conf.mapperMode = pvd::PVRequestMapper::Mask;
                } else if(str=="Slice") {
                    conf.mapperMode = pvd::PVRequestMapper::Slice;
                } else {
                    PyErr_Format(PyExc_ValueError, "Invalid mapperMode %s", str.c_str());
                    return -1;
                }
            }

            // TODO: warn about unknown options?
        }

        if(SELF) {
            // already set by P4PSharedPV_wrap()
        } else if(handler != Py_None) {
            PVHandler::shared_pointer H(new PVHandler(handler));

            SELF = pvas::SharedPV::build(H, &conf);
        } else {
            SELF = pvas::SharedPV::buildReadOnly(&conf);
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

static PyObject* sharedpv_current(PyObject* self) {
    TRY {
        TRACE("");
        pvd::PVStructure::shared_pointer val(SELF->build());
        pvd::BitSet::shared_pointer changed(new pvd::BitSet);

        {
            PyUnlock U;
            SELF->fetch(*val, *changed);
        }
        return P4PValue_wrap(P4PValue_type, val, changed);
    }CATCH()
    return NULL;
}


static PyObject* sharedpv_close(PyObject* self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"destroy", NULL};
        PyObject *destroy = Py_False;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O", (char**)names, &destroy))
            return NULL;

        TRACE("");
        {
            PyUnlock U;
            SELF->close(PyObject_IsTrue(destroy));
        }
        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static PyObject* sharedpv_isOpen(PyObject* self) {
    TRY {
        TRACE("");
        bool opened;

        {
            PyUnlock U;
            opened = SELF->isOpen();
        }
        if(opened)
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
    {"current", (PyCFunction)&sharedpv_current, METH_NOARGS,
     "current() -> Value\n"
     "The current Value of this PV.  The result of the initial value, and all accumulated post() calls."},
    {"close", (PyCFunction)&sharedpv_close, METH_VARARGS|METH_KEYWORDS,
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

PyObject* operation_name(PyObject *self)
{
    TRY {
        return PyUnicode_FromString(SELF.channelName().c_str());
    } CATCH()
    return NULL;
}

PyObject* operation_peer(PyObject *self)
{
    TRY {
        const pva::PeerInfo* info = SELF.peer();
        if(info && !info->peer.empty())
            return PyUnicode_FromString(info->peer.c_str());

        pva::ChannelBaseRequester::shared_pointer req(SELF.getRequester());
        if(req) {
            return PyUnicode_FromString(req->getRequesterName().c_str());
        } else {
            Py_RETURN_NONE;
        }
    } CATCH()
    return NULL;
}

PyObject* operation_account(PyObject *self)
{
    TRY {
        const pva::PeerInfo* info = SELF.peer();

        return PyUnicode_FromString(info ? info->account.c_str() : "");
    } CATCH()
    return NULL;
}

PyObject* operation_roles(PyObject *self)
{
    TRY {
        const pva::PeerInfo* info = SELF.peer();

        PyRef roles(PySet_New(0));

        if(info) {
            for(pva::PeerInfo::roles_t::const_iterator it(info->roles.begin()), end(info->roles.end()); it!=end; ++it) {
                PyRef temp(PyUnicode_FromString(it->c_str()));
                if(PySet_Add(roles.get(), temp.get()))
                    throw std::runtime_error("XXX");
            }

        }

        return roles.release();
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
     "pvRequest() -> Value\n"
     "Access to client provided request modifiers"},
    {"value", (PyCFunction)&operation_value, METH_NOARGS,
     "value() -> Value\n"
     "Access to client provided input/argument value"},
    {"name", (PyCFunction)&operation_name, METH_NOARGS,
     "name() -> str\n"
     "Channel name through which this Operation is made."},
    {"peer", (PyCFunction)&operation_peer, METH_NOARGS,
     "peer() -> str\n"
     "A information about the peer, or None."},
    {"account", (PyCFunction)&operation_account, METH_NOARGS,
     "account() -> str\n"
     "Peer account name, or \"\""},
    {"roles", (PyCFunction)&operation_roles, METH_NOARGS,
     "roles() -> {str}\n"
     "Peer roles, or empty set."},
    {"done", (PyCFunction)&operation_done, METH_VARARGS|METH_KEYWORDS,
     "done(value=None, error=None)\n"
     "Complete in-progress operation.\n"
     "Provide a value=Value (RPC) or value=None (Put) to indicate success."
     "  Provide error=str to signal that an error occured."},
    {"info", (PyCFunction)&operation_info, METH_VARARGS,
     "info(msg)\n"
     "Send remote info message"},
    {"warn", (PyCFunction)&operation_warn, METH_VARARGS,
     "warn(msg)\n"
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
