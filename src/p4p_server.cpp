
#include <sstream>

#include <stddef.h>

#include <pv/serverContext.h>
#include <pv/typeCast.h>

#include "p4p.h"

namespace {

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

struct Server {
    std::string providers;
    pva::Configuration::shared_pointer conf;
    pva::ServerContext::shared_pointer server;
    bool started;
    Server() :started(false) {}
    ~Server() {
        TRACE("ServerContext use_count="<<server.use_count());
        if(server && !server.unique()) {
            std::ostringstream strm;
            strm<<"Server Leaking ServerContext use_count="<<server.use_count();
            PyErr_Warn(PyExc_UserWarning, strm.str().c_str());
        }
        {
            PyUnlock U;
            server.reset();
        }
    }
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
        if(provs) {
            SELF.providers = provs;
            TRACE("Providers: "<<SELF.providers);
        }

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

            B.push_map();
        } else {
            PyErr_Format(PyExc_ValueError, "conf=%s not valid", Py_TYPE(cdict)->tp_name);
            return -1;
        }

        if(!SELF.providers.empty())
            B.add("EPICS_PVAS_PROVIDER_NAMES",SELF.providers).push_map();

        SELF.conf = B.build();

        pva::ServerContext::shared_pointer S(pva::ServerContext::create(pva::ServerContext::Config()
                                                                        .config(SELF.conf)));
        TRACE("ServerContext use_count="<<S.use_count());
        if(!S.unique()) {
            std::ostringstream strm;
            strm<<"ServerContext not unique() after ctor use_count="<<S.use_count();
            PyErr_Warn(PyExc_UserWarning, strm.str().c_str());
        }
        SELF.server = S;

        return 0;
    }CATCH()
    return -1;
}

PyObject* P4PServer_run(PyObject *self)
{
    TRY {
        TRACE("ENTER");

        if(SELF.started) {
            return PyErr_Format(PyExc_RuntimeError, "Already running");
        }
        SELF.started = true;

        pva::ServerContext::shared_pointer S(SELF.server);

        TRACE("UNLOCK");
        {
            PyUnlock U; // release GIL

            S->run(0); // 0 == run forever (unless ->shutdown())
        }
        TRACE("RELOCK");

        SELF.server.reset();

        S->shutdown();

        TRACE("EXIT");
        Py_RETURN_NONE;
    }CATCH()
    TRACE("ERROR");
    return NULL;
}

PyObject* P4PServer_stop(PyObject *self)
{
    TRY {
        if(SELF.server) {
            TRACE("SHUTDOWN");
            PyUnlock U;
            SELF.server->shutdown();
        } else
            TRACE("SKIP");
        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject* P4PServer_conf(PyObject *self)
{

    TRY {
        if(!SELF.server)
            return PyErr_Format(PyExc_RuntimeError, "Server already stopped");

        PyRef ret(PyDict_New());

        pva::Configuration::shared_pointer conf(SELF.server->getCurrentConfig());
        pva::Configuration::keys_t keys(conf->keys());

        for(pva::Configuration::keys_t::const_iterator it = keys.begin(); it!=keys.end(); ++it)
        {
            PyRef val(PyUnicode_FromString(conf->getPropertyAsString(*it, "").c_str()));
            if(PyDict_SetItemString(ret.get(), it->c_str(), val.get()))
                return NULL;
        }

        return ret.release();
    }CATCH()
    return NULL;
}

static PyMethodDef P4PServer_methods[] = {
    {"run", (PyCFunction)&P4PServer_run, METH_NOARGS,
     "Run server (blocking)"},
    {"stop", (PyCFunction)&P4PServer_stop, METH_NOARGS,
     "break from blocking run()"},
    {"conf", (PyCFunction)&P4PServer_conf, METH_NOARGS,
     "conf()\n\n"
     "Return actual Server configuration."},
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
    "p4p._p4p.Server",
    sizeof(P4PServer),
};

} // namespace

void p4p_server_register(PyObject *mod)
{
    P4PServer::buildType();
    P4PServer::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    P4PServer::type.tp_init = &P4PServer_init;
    P4PServer::type.tp_traverse = &P4PServer_traverse;
    P4PServer::type.tp_clear = &P4PServer_clear;

    P4PServer::type.tp_methods = P4PServer_methods;

    if(PyType_Ready(&P4PServer::type))
        throw std::runtime_error("failed to initialize p4p._p4p.Server");

    Py_INCREF((PyObject*)&P4PServer::type);
    if(PyModule_AddObject(mod, "Server", (PyObject*)&P4PServer::type)) {
        Py_DECREF((PyObject*)&P4PServer::type);
        throw std::runtime_error("failed to add p4p._p4p.Server");
    }
}
