
#include <map>

#include <stddef.h>

#include <epicsTime.h>
#include <pv/serverContext.h>
#include <pva/server.h>

#include "p4p.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

typedef PyClassWrapper<pvas::DynamicProvider::shared_pointer, true> PyDynamicProvider;
typedef PyClassWrapper<pvas::StaticProvider::shared_pointer, true> PyStaticProvider;

const unsigned maxCache = 100;
const unsigned expireIn = 5;

PyClassWrapper_DEF(PyDynamicProvider, "DynamicProvider")
PyClassWrapper_DEF(PyStaticProvider, "StaticProvider")

namespace {

struct DynamicHandler : public pvas::DynamicProvider::Handler {
    POINTER_DEFINITIONS(DynamicHandler);

    static size_t num_instances;

    // cache negative search results (names we don't have)
    // map name -> expiration time
    typedef std::map<std::string, epicsTime> search_cache_t;
    search_cache_t search_cache;
    epicsMutex search_cache_lock;

    PyRef cb;
    DynamicHandler(PyObject *callback) :cb(callback, borrow()) {
        REFTRACE_INCREMENT(num_instances);
        TRACE("");
    }
    virtual ~DynamicHandler() {
        // we may get here with the GIL locked (via ~PySharedPV), or not (SharedPV released from ServerContext)
        PyLock L;
        TRACE("");
        cb.reset();
        REFTRACE_DECREMENT(num_instances);
    }

    virtual void hasChannels(pvas::DynamicProvider::search_type& name) {
        epicsTime now(epicsTime::getCurrent());

        for(pvas::DynamicProvider::search_type::iterator it(name.begin()), end(name.end()); it!=end; ++it) {
            TRACE("ENTER "<<it->name());

            bool hit;
            {
                Guard G(search_cache_lock);
                search_cache_t::iterator it2 = search_cache.find(it->name());
                if(it2!=search_cache.end() && now >= it2->second) {
                    // stale entry
                    search_cache.erase(it2);
                    it2 = search_cache.end();
                }
                hit = it2 != search_cache.end();
            }

            if(hit) {
                TRACE("HIT IGNORE "<<it->name());
                continue;
            }


            PyLock L;
            if(cb) {
                PyRef grab(PyObject_CallMethod(cb.get(), "testChannel", "s", it->name().c_str()), allownull());
                if(!grab) {
                    TRACE("cb ERROR");
                    PyErr_Print();
                    PyErr_Clear();

                } else if(PyObject_IsTrue(grab.get())) {
                    TRACE("CLAIM");
                    it->claim();
                    continue;
                } else if(PyBytes_Check(cb.get()) && strcmp(PyBytes_AsString(cb.get()), "nocache")==0) {
                    TRACE("NOCACHE");
                    continue;
                }
            } else {
                TRACE("DEFUCT");
                break;
            }

            // negative

            now += expireIn;

            {
                Guard G(search_cache_lock);
                if(search_cache.size()>=maxCache) {
                    // instead of a proper LRU cache, just drop when it gets too big
                    TRACE("CLEAR IGNORES");
                    search_cache.clear();
                }
                TRACE("ADD IGNORE "<<it->name());
                search_cache.insert(std::make_pair(it->name(), now));
            }
        }
    }

    virtual void listChannels(names_type& names, bool& dynamic) {}

    virtual std::tr1::shared_ptr<epics::pvAccess::Channel> createChannel(const std::tr1::shared_ptr<epics::pvAccess::ChannelProvider>& provider,
                                                                         const std::string& name,
                                                                         const std::tr1::shared_ptr<epics::pvAccess::ChannelRequester>& requester)
    {
        std::tr1::shared_ptr<epics::pvAccess::Channel> ret;
        pvas::SharedPV::shared_pointer pv;

        {
            PyLock G;
            TRACE(cb.get());

            if(cb) {
                PyRef handler(PyObject_CallMethod(cb.get(), "makeChannel", "ss", name.c_str(),
                                                  requester->getRequesterName().c_str()), allownull());
                if(!handler) {
                    TRACE("ERROR");
                    PyErr_Print();
                    PyErr_Clear();

                } else if(!PyObject_IsInstance(handler.get(), (PyObject*)P4PSharedPV_type)) {
                    TRACE("Wrong return type");
                    PyErr_Format(PyExc_TypeError, "makeChannel() must return SharedPV");
                    PyErr_Print();
                    PyErr_Clear();
                }

                pv = P4PSharedPV_unwrap(handler.get());
            }
        }

        if(pv)
            ret = pv->connect(provider, name, requester);
        return ret;
    }

    virtual void destroy() {}
};

size_t DynamicHandler::num_instances;

#define TRY PyDynamicProvider::reference_type SELF = PyDynamicProvider::unwrap(self); try

static int dynamicprovider_init(PyObject *self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"name", "handler", NULL};
        const char *name;
        PyObject *handler;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "sO", (char**)names, &name, &handler))
            return -1;

        DynamicHandler::shared_pointer H(new DynamicHandler(handler));

        SELF.reset(new pvas::DynamicProvider(name, H));

        return 0;
    }CATCH()
    return -1;
}

static int dynamicprovider_traverse(PyObject *self, visitproc visit, void *arg)
{
    TRY {
        if(!SELF) return 0; // eg. failed sub-class ctor
        // attempt to drill through to the handler
        DynamicHandler::shared_pointer handler(std::tr1::dynamic_pointer_cast<DynamicHandler>(SELF->getHandler()));
        if(handler && handler->cb)
            Py_VISIT(handler->cb.get());
        return 0;
    } CATCH()
    return -1;
}

static int dynamicprovider_clear(PyObject *self)
{
    TRY {
        // also called by PyClassWrapper dtor, so we are killing the Handler
        // even though it may still be ref'd
        TRACE("");
        if(!SELF) return 0; // eg. failed sub-class ctor
        DynamicHandler::shared_pointer handler(std::tr1::dynamic_pointer_cast<DynamicHandler>(SELF->getHandler()));
        // ~= Py_CLEAR(cb)
        if(handler) {
            PyRef tmp;
            handler->cb.swap(tmp);
        }
        return 0;
    } CATCH()
    return -1;
}

#undef TRY
#define TRY PyStaticProvider::reference_type SELF = PyStaticProvider::unwrap(self); try

static int staticprovider_init(PyObject *self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"name", NULL};
        const char *name;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "s", (char**)names, &name))
            return -1;

        SELF.reset(new pvas::StaticProvider(name));

        return 0;
    }CATCH()
    return -1;
}

static PyObject* staticprovider_close(PyObject *self) {
    TRY {
        {
            PyUnlock U;
            SELF->close();
        }

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static PyObject* staticprovider_add(PyObject *self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"name", "pv", NULL};
        const char *name;
        PyObject *pypv;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "sO", (char**)names, &name, &pypv))
            return NULL;

        if(PyObject_IsInstance(pypv, (PyObject*)P4PSharedPV_type)) {

            {
                PyUnlock U;
                SELF->add(name, P4PSharedPV_unwrap(pypv));
            }

        } else {
            return PyErr_Format(PyExc_ValueError, "pv= must be SharedPV instance");
        }

        Py_RETURN_NONE;
    }CATCH()
            return NULL;
}

static PyObject* staticprovider_remove(PyObject *self, PyObject *args, PyObject *kwds) {
    TRY {
        const char *names[] = {"name", NULL};
        const char *name;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "s", (char**)names, &name))
            return NULL;

        pvas::SharedPV::shared_pointer sharedpv;
        pvas::StaticProvider::ChannelBuilder::shared_pointer pv;
        {
            PyUnlock U;
            pv = SELF->remove(name);
        }

        if(!pv) {
            return PyErr_Format(PyExc_KeyError, "No Such PV %s", name);

        } else if((sharedpv = std::tr1::dynamic_pointer_cast<pvas::SharedPV>(pv))) {
            return P4PSharedPV_wrap(sharedpv);

        } else {
            return PyErr_Format(PyExc_TypeError, "PV %s of unmapped c++ type", name);
        }

        Py_RETURN_NONE; // not reached
    }CATCH()
    return NULL;
}

static PyObject* staticprovider_keys(PyObject *self)
{
    TRY {
        PyRef ret(PyList_New(0));

        for(pvas::StaticProvider::const_iterator it(SELF->begin()), end(SELF->end()); it!=end; ++it)
        {
            PyRef name(PyUnicode_FromString(it->first.c_str()));
            if(PyList_Append(ret.get(), name.get()))
                return NULL;
        }

        return ret.release();
    }CATCH()
    return NULL;
}

static PyMethodDef StaticProvider_methods[] = {
    {"close", (PyCFunction)&staticprovider_close, METH_NOARGS,
     "close()\n"
     "Equivalent to calling `SharedPV.close()` on every SharedPV `add()`'d"},
    {"add", (PyCFunction)&staticprovider_add, METH_VARARGS|METH_KEYWORDS,
     "add(name, pv)\n"
     "Add a new SharedPV instance to be served by this provider."},
    {"remove", (PyCFunction)&staticprovider_remove, METH_VARARGS|METH_KEYWORDS,
     "remove(name) -> pv\n"
     "Remove a SharedPV from this provider.  Raises KeyError if named PV doesn't exist.\n"
     "Returns the PV which has been removed.\n"
     "Implicitly `close()` s the PV before returning, disconnecting any clients."},
    {"keys", (PyCFunction)&staticprovider_keys, METH_NOARGS,
     "keys() -> [name]\n"
     "Returns a list of PV names\n"},
    {NULL}
};


} // namespace

epics::pvAccess::ChannelProvider::shared_pointer p4p_unwrap_provider(PyObject *provider)
{
    if(PyObject_IsInstance(provider, (PyObject*)&PyDynamicProvider::type))
        return PyDynamicProvider::unwrap(provider)->provider();
    else if(PyObject_IsInstance(provider, (PyObject*)&PyStaticProvider::type))
        return PyStaticProvider::unwrap(provider)->provider();
    else
        throw std::runtime_error("provider= must be DynamicProvider or StaticProvider");
}

PyObject* p4p_add_provider(PyObject *junk, PyObject *args, PyObject *kwds)
{
    const char *name;
    PyObject *prov;

    const char *names[] = {"name", "provider", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "sO", (char**)names, &name, &prov))
        return NULL;

    try {
        pva::ChannelProvider::shared_pointer provider(p4p_unwrap_provider(prov));
        // TODO: avoid redundency
        if(provider->getProviderName()!=name)
            return PyErr_Format(PyExc_ValueError, "Provider name inconsistent %s != %s",
                                provider->getProviderName().c_str(), name);
        TRACE("Add "<<provider->getProviderName());
        pva::ChannelProviderRegistry::servers()->addSingleton(provider);

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject* p4p_remove_provider(PyObject *junk, PyObject *args, PyObject *kwds)
{
    const char *name;
    const char *names[] = {"name", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "s", (char**)names, &name))
        return NULL;

    try {
        TRACE("Clear "<<name);

        pva::ChannelProviderRegistry::servers()->remove(name);

        Py_RETURN_TRUE;
    }CATCH()
    return NULL;
}

PyObject* p4p_remove_all(PyObject *junk, PyObject *args, PyObject *kwds)
{
    const char *names[] = {NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "", (char**)names))
        return NULL;

    try {
        TRACE("Clear");

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

void p4p_server_provider_register(PyObject *mod)
{
    PyDynamicProvider::buildType();
    PyDynamicProvider::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC;
    PyDynamicProvider::type.tp_init = &dynamicprovider_init;
    PyDynamicProvider::type.tp_traverse = &dynamicprovider_traverse;
    PyDynamicProvider::type.tp_clear = &dynamicprovider_clear;

    PyDynamicProvider::finishType(mod, "DynamicProvider");

    PyStaticProvider::buildType();
    PyStaticProvider::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyStaticProvider::type.tp_init = &staticprovider_init;
    // we are explicitly not GC'ing, through we could traverse through PySharedPV to it's handlers

    PyStaticProvider::type.tp_methods = StaticProvider_methods;

    PyStaticProvider::finishType(mod, "StaticProvider");

    epics::registerRefCounter("p4p._p4p.DynamicProvider::Handler", &DynamicHandler::num_instances);
}
