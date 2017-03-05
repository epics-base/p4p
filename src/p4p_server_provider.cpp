
#include <stddef.h>

#include <pv/serverContext.h>

#include "p4p.h"

namespace {

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

struct PyServerChannel;

struct PyServerProvider :
        public pva::ChannelProviderFactory,
        public pva::ChannelFind,
        public pva::ChannelProvider,
        public std::tr1::enable_shared_from_this<PyServerProvider>
{
    POINTER_DEFINITIONS(PyServerProvider);

    virtual ~PyServerProvider() {}

    PyExternalRef provider;
    std::string provider_name;

    virtual std::string getFactoryName() { return provider_name; }
    virtual ChannelProvider::shared_pointer sharedInstance() {
        return shared_from_this();
    }

    virtual std::tr1::shared_ptr<pva::ChannelProvider> getChannelProvider() { return shared_from_this(); }
    virtual void cancel() {}

    virtual void lock() {}
    virtual void unlock() {}
    virtual void destroy() {}

    virtual std::string getProviderName() { return provider_name; }

    virtual pva::ChannelFind::shared_pointer channelFind(std::string const & channelName,
            pva::ChannelFindRequester::shared_pointer const & channelFindRequester)
    {
        TRACE("ENTER "<<channelName);
        pva::ChannelFind::shared_pointer ret;
        try {
            // TODO: cache results for a short time to reduce calls

            PyLock G;

            PyRef grab(PyObject_CallMethod(provider.ref.get(), "testChannel", "s", channelName.c_str()), allownull());
            if(!grab.get()) {
                PyErr_Print();
                PyErr_Clear();
                channelFindRequester->channelFindResult(pvd::Status(pvd::Status::STATUSTYPE_ERROR, "Logic Error"),
                                                        ret, false);

            } else if(PyObject_IsTrue(grab.get())) {
                ret = shared_from_this();
                channelFindRequester->channelFindResult(pvd::Status::Ok,
                                                        ret, true);

            } else {
                channelFindRequester->channelFindResult(pvd::Status::Ok,
                                                        ret, false);
            }
        } catch(std::exception& e) {
            channelFindRequester->channelFindResult(pvd::Status(pvd::Status::STATUSTYPE_ERROR, e.what()),
                                                    ret, false);
        }
        TRACE("EXIT "<<(ret ? "Claim" : "Ignore"));
        return ret;
    }

    virtual pva::ChannelFind::shared_pointer channelList(pva::ChannelListRequester::shared_pointer const & channelListRequester)
    {
        pva::ChannelFind::shared_pointer ret;
        channelListRequester->channelListResult(pvd::Status(pvd::Status::STATUSTYPE_FATAL, "Not implemented"),
                                                ret,
                                                pvd::PVStringArray::const_svector(),
                                                false);
        return ret;
    }

    virtual pva::Channel::shared_pointer createChannel(std::string const & channelName,
                                                       pva::ChannelRequester::shared_pointer const & channelRequester,
                                                       short priority)
    {
        return createChannel(channelName, channelRequester, priority, "<unknown>");
    }

    virtual pva::Channel::shared_pointer createChannel(std::string const & channelName,
                                                       pva::ChannelRequester::shared_pointer const & channelRequester,
                                                       short priority, std::string const & address);
};

struct PyServerChannel :
        public pva::Channel,
        public std::tr1::enable_shared_from_this<PyServerChannel>
{
    POINTER_DEFINITIONS(PyServerChannel);

    PyServerProvider::shared_pointer provider;
    pva::ChannelRequester::shared_pointer requester;
    const std::string name;
    PyExternalRef handler;

    PyServerChannel(const PyServerProvider::shared_pointer& provider,
                    const pva::ChannelRequester::shared_pointer& req,
                    const std::string& name,
                    PyRef& py)
        :provider(provider)
        ,requester(req)
        ,name(name)
    {
        handler.swap(py);
    }
    virtual ~PyServerChannel() {}

    virtual void destroy() {}

    virtual std::tr1::shared_ptr<pva::ChannelProvider> getProvider() { return provider; }
    virtual std::string getRemoteAddress() { return requester->getRequesterName(); }
    virtual ConnectionState getConnectionState() { return pva::Channel::CONNECTED; }
    virtual std::string getChannelName() { return name; }
    virtual std::tr1::shared_ptr<pva::ChannelRequester> getChannelRequester() { return requester; }

    virtual void getField(pva::GetFieldRequester::shared_pointer const & requester,std::string const & subField)
    {
        requester->getDone(pvd::Status(pvd::Status::STATUSTYPE_FATAL, "Not implemented"),
                           pvd::FieldConstPtr());
    }

    virtual pva::AccessRights getAccessRights(epics::pvData::PVField::shared_pointer const & pvField)
    {
        return pva::readWrite;
    }

    virtual pva::ChannelRPC::shared_pointer createChannelRPC(
            pva::ChannelRPCRequester::shared_pointer const & channelRPCRequester,
            pvd::PVStructure::shared_pointer const & pvRequest);
};

struct PyServerRPC : public pva::ChannelRPC,
                     public std::tr1::enable_shared_from_this<PyServerRPC>
{
    POINTER_DEFINITIONS(PyServerRPC);

    struct ReplyData {
        PyServerRPC::shared_pointer rpc;
        bool sent;
        ReplyData() :sent(false) {}
    };

    typedef PyClassWrapper<ReplyData> Reply;

    PyServerChannel::weak_pointer chan;
    pva::ChannelRPCRequester::weak_pointer requester;
    bool inprog;

    PyServerRPC(const PyServerChannel::shared_pointer& c,
                const pva::ChannelRPCRequester::shared_pointer& r) :chan(c), requester(r), inprog(false) {}
    virtual ~PyServerRPC() {}

    virtual void lock() {}
    virtual void unlock() {}
    virtual void destroy() {}

    virtual pva::Channel::shared_pointer getChannel() { return pva::Channel::shared_pointer(chan); }

    virtual void cancel() {}
    virtual void lastRequest() {}

    virtual void request(pvd::PVStructure::shared_pointer const & pvArgument)
    {
        TRACE("ENTER");
        PyServerChannel::shared_pointer C(chan.lock());
        pva::ChannelRPCRequester::shared_pointer R(requester.lock());
        if(!C || !R) return;

        bool createdReply = false;
        PyLock G;
        try {

            PyRef wrapper(PyObject_GetAttrString(C->handler.ref.get(), "Value"));
            if(!PyType_Check(wrapper.get()))
                throw std::runtime_error("handler.Value is not a Type");

            PyRef req(P4PValue_wrap((PyTypeObject*)wrapper.get(), pvArgument));

            PyRef rep(Reply::type.tp_alloc(&Reply::type, 0));
            createdReply = true;
            // from this point the Reply is responsible for calling requestDone()
            // if the call to .rpc() fails then the reply is sent when the Reply is destroyed
            // which means that an exception thrown by rpc() will only be printed to screen

            Reply::unwrap(rep.get()).rpc = shared_from_this();

            PyRef junk(PyObject_CallMethod(C->handler.ref.get(), "rpc", "OO", rep.get(), req.get()));

            TRACE("SUCCESS");
        } catch(std::exception& e) {
            if(PyErr_Occurred()) {
                PyErr_Print();
                PyErr_Clear();
            }
            if(!createdReply)
                R->requestDone(pvd::Status(pvd::Status::STATUSTYPE_ERROR, e.what()),
                               shared_from_this(),
                               pvd::PVStructurePtr());
            TRACE("ERROR "<<(createdReply ? "DELEGATE" : "SENT"));
        }
    }

    static void reply_dealloc(PyObject *raw) {
        try {
            TRACE("ENTER");
            Reply::reference_type SELF = Reply::unwrap(raw);
            if(!SELF.sent) {
                TRACE("SEND ERROR");
                pva::ChannelRPCRequester::shared_pointer R(SELF.rpc->requester.lock());
                if(R)
                    R->requestDone(pvd::Status(pvd::Status::STATUSTYPE_ERROR, "No Reply"),
                                   SELF.rpc,
                                   pvd::PVStructurePtr());
            }
        } catch(std::exception& e) {
            std::cerr<<"Error in RPC reply dtor "<<e.what()<<"\n";
        }
        Reply::tp_dealloc(raw);
    }

    static PyObject* reply_done(PyObject *self, PyObject *args, PyObject *kwds)
    {
        PyObject *data = Py_None;
        const char *error = NULL;
        const char *names[] = {"reply", "error", NULL};
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|Oz", (char**)names,
                                        &data,
                                        &error))
            return NULL;

        TRACE("ENTER");
        Reply::reference_type SELF = Reply::unwrap(self);
        try {
            pva::ChannelRPCRequester::shared_pointer R(SELF.rpc->requester.lock());

            if(!R) {
                // requester is gone (server lost connection)
                // no-op

            } else if(SELF.sent) {
                return PyErr_Format(PyExc_TypeError, "done() already called");

            } else if(data!=Py_None) {


                pvd::PVStructure::shared_pointer value;

                if(PyObject_TypeCheck(data, P4PValue_type)) {
                    value = P4PValue_unwrap(data);
                } else {
                    return PyErr_Format(PyExc_ValueError, "RPC results must be Value");
                }

                R->requestDone(pvd::Status::Ok,
                               SELF.rpc,
                               value);
                SELF.sent = true;
            } else if(error) {
                R->requestDone(pvd::Status(pvd::Status::STATUSTYPE_ERROR, error),
                                       SELF.rpc,
                                       pvd::PVStructurePtr());
                SELF.sent = true;
            } else {
                return PyErr_Format(PyExc_ValueError, "done() needs reply= or error=");
            }

            TRACE("SUCCESS");
            Py_RETURN_NONE;
        }CATCH()
        TRACE("ERROR");
        return NULL;
    }
};

pva::Channel::shared_pointer
PyServerProvider::createChannel(std::string const & channelName,
                                                   pva::ChannelRequester::shared_pointer const & channelRequester,
                                                   short priority, std::string const & address)
{
    TRACE("ENTER "<<channelRequester->getRequesterName());
    pva::Channel::shared_pointer ret;
    try {
        PyLock G;

        PyRef handler(PyObject_CallMethod(provider.ref.get(), "makeChannel", "ss", channelName.c_str(),
                                      channelRequester->getRequesterName().c_str()), allownull());
        if(!handler.get()) {
            PyErr_Print();
            PyErr_Clear();
            channelRequester->channelCreated(pvd::Status(pvd::Status::STATUSTYPE_ERROR, "Logic Error"),
                                             ret);

        } else if(handler.get()==Py_None) {
            channelRequester->channelCreated(pvd::Status(pvd::Status::STATUSTYPE_ERROR, "No such channel"),
                                             ret);

        } else {
            ret.reset(new PyServerChannel(shared_from_this(),channelRequester, channelName, handler));
            // handler consumed now
            channelRequester->channelCreated(pvd::Status::Ok, ret);
        }
    } catch(std::exception& e) {
        channelRequester->channelCreated(pvd::Status(pvd::Status::STATUSTYPE_ERROR, e.what()),
                                         ret);
    }
    TRACE("EXIT "<<(ret ? "Create" : "Refuse"));
    return ret;
}

pva::ChannelRPC::shared_pointer
PyServerChannel::createChannelRPC(
        pva::ChannelRPCRequester::shared_pointer const & channelRPCRequester,
        pvd::PVStructure::shared_pointer const & pvRequest)
{
    TRACE("ENTER");
    // TODO: pvRequest being ignored
    pva::ChannelRPC::shared_pointer ret(new PyServerRPC(shared_from_this(), channelRPCRequester));
    channelRPCRequester->channelRPCConnect(pvd::Status::Ok, ret);
    return ret;
}

typedef std::map<std::string, PyServerProvider::shared_pointer> pyproviders_t;
pyproviders_t* pyproviders;

PyObject* p4p_add_provider(PyObject *junk, PyObject *args, PyObject *kwds)
{
    const char *name;
    PyObject *prov;

    const char *names[] = {"name", "provider", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "sO", (char**)names, &name, &prov))
        return NULL;

    try {
        if(!pyproviders)
            pyproviders = new pyproviders_t;

        pyproviders_t::const_iterator it = pyproviders->find(name);
        if(it!=pyproviders->end())
            return PyErr_Format(PyExc_KeyError, "Provider %s already registered", name);

        PyRef handler(prov, borrow());
        PyServerProvider::shared_pointer P(new PyServerProvider);
        P->provider.swap(handler);
        P->provider_name = name;

        pva::registerChannelProviderFactory(P);

        (*pyproviders)[name] = P;
        TRACE("name="<<name);

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
        if(!pyproviders)
            return PyErr_Format(PyExc_KeyError, "Provider %s not registered", name);

        pyproviders_t::iterator it = pyproviders->find(name);
        if(it==pyproviders->end())
            return PyErr_Format(PyExc_KeyError, "Provider %s not registered", name);

        pva::unregisterChannelProviderFactory(it->second);

        PyRef X;
        it->second->provider.swap(X);

        pyproviders->erase(it);
        TRACE("name="<<name);

        Py_RETURN_NONE;
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
        if(pyproviders) delete pyproviders;

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

static struct PyMethodDef PyServerRPC_methods[] = {
    {"done", (PyCFunction)PyServerRPC::reply_done, METH_VARARGS|METH_KEYWORDS,
     "Complete RPC call with reply data or error message"},
    {NULL}
};

int PyServerRPC_traverse(PyObject *self, visitproc visit, void *arg)
{
    return 0;
}

int PyServerRPC_clear(PyObject *self)
{
    return 0;
}

template<>
PyTypeObject PyServerRPC::Reply::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_p4p.RPCReply",
    sizeof(PyServerRPC::Reply),
};

} // namespace

struct PyMethodDef P4P_methods[] = {
    {"installProvider", (PyCFunction)p4p_add_provider, METH_VARARGS|METH_KEYWORDS,
     "Install a new Server Channel provider"},
    {"removeProvider", (PyCFunction)p4p_remove_provider, METH_VARARGS|METH_KEYWORDS,
     "Remove a previously added Server Channel provider"},
    {"clearProviders", (PyCFunction)p4p_remove_all, METH_VARARGS|METH_KEYWORDS,
     "Remove all Server Channel providers"},
    {NULL}
};

void p4p_server_provider_register(PyObject *mod)
{
    pyproviders = NULL;

    PyServerRPC::Reply::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    // No new or init.  This type can't be created by python code
    PyServerRPC::Reply::type.tp_dealloc = &PyServerRPC::reply_dealloc;
    PyServerRPC::Reply::type.tp_traverse = &PyServerRPC_traverse;
    PyServerRPC::Reply::type.tp_clear = &PyServerRPC_clear;

    PyServerRPC::Reply::type.tp_methods = PyServerRPC_methods;

    PyServerRPC::Reply::type.tp_weaklistoffset = offsetof(PyServerRPC::Reply, weak);

    if(PyType_Ready(&PyServerRPC::Reply::type))
        throw std::runtime_error("failed to initialize RPCReply");

    Py_INCREF((PyObject*)&PyServerRPC::Reply::type);
    if(PyModule_AddObject(mod, "RPCReply", (PyObject*)&PyServerRPC::Reply::type)) {
        Py_DECREF((PyObject*)&PyServerRPC::Reply::type);
        throw std::runtime_error("failed to add _p4p.RPCReply");
    }
}
