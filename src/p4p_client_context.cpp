
#include <map>
#include <set>
#include <list>
#include <iostream>
#include <typeinfo>

#include <stdlib.h>

#include <epicsMutex.h>
#include <epicsGuard.h>

#include <pv/pvAccess.h>
#include <pv/logger.h>
#include <pv/clientFactory.h>
#include <pv/caProvider.h>
#include <pv/configuration.h>

#include "p4p.h"
#include "p4p_client.h"

namespace {

typedef epicsGuard<epicsMutex> Guard;

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

struct Context {
    POINTER_DEFINITIONS(Context);

    // TODO: visit() and see if this is a PyServerProvider instance
    pva::ChannelProvider::shared_pointer provider;

    char *name;

    Context() :name(NULL) {}
    ~Context() { close(); free(name); }

    void close();

    static int       py_init(PyObject *self, PyObject *args, PyObject *kws);
    static PyObject *py_channel(PyObject *self, PyObject *args, PyObject *kws);
    static PyObject *py_close(PyObject *self);

    static PyObject *py_providers(PyObject *junk);
    static PyObject *py_set_debug(PyObject *junk, PyObject *args, PyObject *kws);
    static PyObject *py_makeRequest(PyObject *junk, PyObject *args);
};

typedef PyClassWrapper<Context> PyContext;

#define TRY PyContext::reference_type SELF = PyContext::unwrap(self); try

int Context::py_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"provider", "conf", "useenv", NULL};
        const char *pname;
        PyObject *cdict = Py_None, *useenv = Py_True;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "s|OO", (char**)names, &pname, &cdict, &useenv))
            return -1;

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

        // we create our own provider.
        // we are greedy and don't want to share (also we can destroy channels at will)
#if 0
        // No way to apply custom config :P
        SELF.provider = pva::getChannelProviderRegistry()->createProvider(pname);
#else
        SELF.provider = pva::getChannelProviderRegistry()->createProvider(pname, B.build());
#endif

        TRACE("Context init");

        if(!SELF.provider)
            throw std::logic_error(SB()<<"No such provider \""<<pname<<"\"");

        SELF.name = strdup(pname);

        return 0;
    } CATCH()
    return -1;
}

PyObject *Context::py_channel(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"channel", NULL};
        char *cname;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "s", (char**)names, &cname))
            return NULL;

        if(!SELF.provider)
            return PyErr_Format(PyExc_RuntimeError, "Context has been closed");

        PyRef klass(PyObject_GetAttrString(self, "Channel"));
        if(!PyType_Check(klass.get()) || !PyType_IsSubtype((PyTypeObject*)klass.get(), &PyChannel::type))
            return PyErr_Format(PyExc_RuntimeError, "self.Channel not valid");
        PyTypeObject *chanklass = (PyTypeObject*)klass.get();

        Channel::shared_pointer pychan(new Channel);
        Channel::Req::shared_pointer pyreq(new Channel::Req(pychan));

        pva::Channel::shared_pointer chan;

        std::string chanName(cname);

        {
            PyUnlock U;
            chan = SELF.provider->createChannel(chanName, pyreq);
        }

        if(!chan) {
            return PyErr_Format(PyExc_RuntimeError, "Failed to create channel '%s'", cname);
        } else if(!chan.unique()) {
            std::cerr<<"Provider "<<chan->getProvider()->getProviderName()
                     <<" for "<<chan->getChannelName()
                     <<" gives non-unique Channel.  use_count="<<chan.use_count()<<"\n";
        }

        pychan->channel = chan;

        PyRef ret(chanklass->tp_new(chanklass, args, kws));

        PyChannel::unwrap(ret.get()).swap(pychan);

        if(chanklass->tp_init && chanklass->tp_init(ret.get(), args, kws))
            throw std::runtime_error("Error Channel.__init__");

        TRACE("Channel "<<cname<<" "<<chan);
        return ret.release();
    } CATCH()
    return NULL;
}

void Context::close()
{
    TRACE("Context close");
    if(provider) {
        PyUnlock U;
        provider.reset();
    }
}

PyObject *Context::py_close(PyObject *self)
{
    TRY {
        SELF.close();
        Py_RETURN_NONE;
    } CATCH()
    return NULL;
}

PyObject*  Context::py_providers(PyObject *junk)
{
    try {
        std::auto_ptr<pva::ChannelProviderRegistry::stringVector_t> names(pva::getChannelProviderRegistry()->getProviderNames());

        if(!names.get())
            return PyErr_Format(PyExc_RuntimeError, "Unable for fetch provider names!?!");

        PyRef ret(PyList_New(names->size()));

        for(size_t i=0; i<names->size(); i++) {
            PyRef name(PyUnicode_FromString((*names)[i].c_str()));

            PyList_SET_ITEM(ret.get(), i, name.release());
        }

        return ret.release();
    }CATCH()
    return NULL;
}

PyObject*  Context::py_set_debug(PyObject *junk, PyObject *args, PyObject *kws)
{
    try {
        int lvl = pva::logLevelError;
        static const char* names[] = {"level", NULL};
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|i", (char**)&names, &lvl))
            return NULL;

        pva::pvAccessSetLogLevel((pva::pvAccessLogLevel)lvl);

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject* Context::py_makeRequest(PyObject *junk, PyObject *args)
{
    try {
        const char *req;
        if(!PyArg_ParseTuple(args, "s", &req))
            return NULL;

        // OMG, would a simple function w/ 'throw' be so much to ask for!?!?!
        pvd::CreateRequest::shared_pointer create(pvd::CreateRequest::create());
        pvd::PVStructure::shared_pointer str(create->createRequest(req));
        if(!str)
            throw std::runtime_error(SB()<<"Error parsing pvRequest: "<<create->getMessage());

        PyRef ret(P4PValue_wrap(P4PValue_type, str));

        return ret.release();
    }CATCH()
    return NULL;
}

static PyMethodDef Context_methods[] = {
    {"channel", (PyCFunction)&Context::py_channel, METH_VARARGS|METH_KEYWORDS,
     "Return a Channel"},
    {"close", (PyCFunction)&Context::py_close, METH_NOARGS,
     "Close this Context"},
    {"providers", (PyCFunction)&Context::py_providers, METH_NOARGS|METH_STATIC,
     "providers() -> ['name', ...]\n"
     ":returns: A list of all currently registered provider names.\n\n"
     "A staticmethod."},
    {"set_debug", (PyCFunction)&Context::py_set_debug, METH_VARARGS|METH_KEYWORDS|METH_STATIC,
     "set_debug(lvl)\n\n"
     "Set PVA debug level"},
    {"makeRequest", (PyCFunction)&Context::py_makeRequest, METH_VARARGS|METH_STATIC,
     "makeRequest(\"field(value)\")\n\nParse pvRequest string"},
    {NULL}
};

static PyMemberDef Context_members[] = {
    {"name", T_STRING, offsetof(PyContext,I)+offsetof(Context, name), READONLY, "Provider name"},
    {NULL}
};

template<>
PyTypeObject PyContext::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Context",
    sizeof(PyContext),
};


void unfactory()
{
    //pva::ca::CAClientFactory::stop();
    pva::ClientFactory::stop();
}

} // namespace

void p4p_client_context_register(PyObject *mod)
{
    pva::ClientFactory::start();
    // don't enable the "ca" provider as it doesn't follow the ChannelProvider rules
    //pva::ca::CAClientFactory::start();

    Py_AtExit(&unfactory);

    PyContext::buildType();
    PyContext::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyContext::type.tp_init = &Context::py_init;

    PyContext::type.tp_methods = Context_methods;
    PyContext::type.tp_members = Context_members;

    if(PyType_Ready(&PyContext::type))
        throw std::runtime_error("failed to initialize PyContext");

    Py_INCREF((PyObject*)&PyContext::type);
    if(PyModule_AddObject(mod, "Context", (PyObject*)&PyContext::type)) {
        Py_DECREF((PyObject*)&PyContext::type);
        throw std::runtime_error("failed to add p4p._p4p.Context");
    }
}
