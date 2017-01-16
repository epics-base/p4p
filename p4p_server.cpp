
#include <stddef.h>

#include <pv/serverContext.h>

#include "p4p.h"

namespace {

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

struct Server {
    std::string providers;
    pva::Configuration::shared_pointer conf;
    pva::ServerContextImpl::shared_pointer server;
};

typedef PyClassWrapper<Server> P4PServer;

#define TRY P4PServer::reference_type SELF = P4PServer::unwrap(self); try

int P4PServer_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *cdict = Py_None, *useenv = Py_True;
    const char *provs = NULL;
    const char *names[] = {"conf", "useenv", "providers", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|OOz", (char**)names, &cdict, &useenv, &provs))
        return -1;

    TRY {
        if(provs)
            SELF.providers = provs;

        pva::ConfigurationBuilder B;

        if(PyObject_IsTrue(useenv))
            B.push_env();

        if(cdict==Py_None) {
            // nothing
        } else if(PyDict_Check(cdict)) {
            Py_ssize_t I = 0;
            PyObject *key, *value;

            while(PyDict_Next(cdict, &I, &key, &value)) {
                PyString K(key), V(value);

                B.add(K.str(), V.str());
            }
        } else {
            PyErr_Format(PyExc_ValueError, "conf=%s not valid", Py_TYPE(cdict)->tp_name);
            return -1;
        }

        SELF.conf = B.build();

        return 0;
    }CATCH()
    return -1;
}

PyObject* P4PServer_run(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char *names[] = {NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "", (char**)names))
        return NULL;

    TRY {
        if(SELF.server) {
            return PyErr_Format(PyExc_RuntimeError, "Already running");
        }

        pva::ServerContextImpl::shared_pointer S(pva::ServerContextImpl::create(SELF.conf));

        if(!SELF.providers.empty())
            S->setChannelProviderName(SELF.providers);

        S->initialize(pva::getChannelProviderRegistry());

        SELF.server = S;

        {
            PyUnlock U; // release GIL

            S->run(0); // 0 == run forever (unless ->shutdown())
        }

        SELF.server.reset();

        S->destroy();

    }CATCH()
    return NULL;
}

PyObject* P4PServer_stop(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char *names[] = {NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "", (char**)names))
        return NULL;

    TRY {
        if(SELF.server) {
            SELF.server->shutdown();
        }
    }CATCH()
    return NULL;
}

static PyMethodDef P4PServer_methods[] = {
    {"run", (PyCFunction)&P4PServer_run, METH_VARARGS|METH_KEYWORDS,
     "Run server (blocking)"},
    {"stop", (PyCFunction)&P4PServer_stop, METH_VARARGS|METH_KEYWORDS,
     "break from blocking run()"},
    {NULL}
};

int P4PServer_traverse(PyObject *self, visitproc visit, void *arg)
{
    return 0;
}

int P4PServer_clear(PyObject *self)
{
    return 0;
}

template<>
PyTypeObject P4PServer::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_p4p._Server",
    sizeof(P4PServer),
};

} // namespace

void p4p_server_register(PyObject *mod)
{
    P4PServer::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    P4PServer::type.tp_new = &P4PServer::tp_new;
    P4PServer::type.tp_init = &P4PServer_init;
    P4PServer::type.tp_dealloc = &P4PServer::tp_dealloc;
    P4PServer::type.tp_traverse = &P4PServer_traverse;
    P4PServer::type.tp_clear = &P4PServer_clear;

    P4PServer::type.tp_methods = P4PServer_methods;

    P4PServer::type.tp_weaklistoffset = offsetof(P4PServer, weak);

    if(PyType_Ready(&P4PServer::type))
        throw std::runtime_error("failed to initialize P4PServer_type");

    Py_INCREF((PyObject*)&P4PServer::type);
    if(PyModule_AddObject(mod, "Type", (PyObject*)&P4PServer::type)) {
        Py_DECREF((PyObject*)&P4PServer::type);
        throw std::runtime_error("failed to add _p4p.Type");
    }
}
