
#include <map>
#include <set>
#include <iostream>

#include <epicsMutex.h>
#include <epicsGuard.h>

#include <pv/pvAccess.h>

#include "p4p.h"

namespace {

typedef epicsGuard<epicsMutex> Guard;

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

struct Context;
struct Channel;
struct OpBase;

struct Context {
    POINTER_DEFINITIONS(Context);

    pva::ChannelProvider::shared_pointer provider;
    typedef std::map<std::string, pva::Channel::shared_pointer> channels_t;
    channels_t channels;

    Context() {}
    ~Context() {}

    static int       py_init(PyObject *self, PyObject *args, PyObject *kws);
    static PyObject *py_channel(PyObject *self, PyObject *args, PyObject *kws);
    static PyObject *py_close(PyObject *self);

    static PyObject *py_providers(PyObject *junk);
};

struct Channel : public pva::ChannelRequester {
    POINTER_DEFINITIONS(Channel);

    Context::shared_pointer context; // Context
    pva::Channel::shared_pointer channel;

    typedef std::set<std::tr1::shared_ptr<OpBase> > operations_t;
    operations_t ops;

    Channel() {}
    virtual ~Channel() {}

    virtual std::string getRequesterName() { return "p4p.Channel"; }

    virtual void channelCreated(const pvd::Status& status, pva::Channel::shared_pointer const & channel);
    virtual void channelStateChange(pva::Channel::shared_pointer const & channel, pva::Channel::ConnectionState connectionState);

    static PyObject *py_get(PyObject *self, PyObject *args, PyObject *kws);
};

struct OpBase {
    POINTER_DEFINITIONS(OpBase);

    Channel::shared_pointer channel; // Channel
    pvd::PVStructure::shared_pointer req;

    OpBase(const Channel::shared_pointer& ch) :channel(ch) {}
    virtual ~OpBase() { cancel(); }

    // pva::Channel life-cycle callbacks
    //  called to (re)start operation
    virtual void restart(const OpBase::shared_pointer& self) {}
    //  channel lost connection
    virtual void lostConn(const OpBase::shared_pointer& self) {}
    //  channel destoryed or user cancel
    virtual bool cancel() {return false;}

    static PyObject *py_cancel(PyObject *self);
};

typedef PyClassWrapper<Context::shared_pointer> PyContext;
typedef PyClassWrapper<Channel::shared_pointer> PyChannel;
typedef PyClassWrapper<OpBase::shared_pointer > PyOp;

struct GetOp : public OpBase, public pva::ChannelGetRequester {
    POINTER_DEFINITIONS(GetOp);

    pva::ChannelGet::shared_pointer op;
    PyRef cb;

    GetOp(const Channel::shared_pointer& ch) :OpBase(ch) {}
    virtual ~GetOp() {}

    virtual void restart(const GetOp::shared_pointer &self);
    virtual void lostConn(const GetOp::shared_pointer& self);
    virtual bool cancel();

    virtual std::string getRequesterName() { return "p4p.Op"; }

    virtual void channelGetConnect(
        const pvd::Status& status,
        pva::ChannelGet::shared_pointer const & channelGet,
        pvd::Structure::const_shared_pointer const & structure);

    virtual void getDone(
        const pvd::Status& status,
        pva::ChannelGet::shared_pointer const & channelGet,
        pvd::PVStructure::shared_pointer const & pvStructure,
        pvd::BitSet::shared_pointer const & bitSet);
};


#define TRY PyContext::reference_type SELF = PyContext::unwrap(self); try


int Context::py_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"provider", NULL};
        const char *pname;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "s", (char**)names, &pname))
            return -1;

        Context::shared_pointer ctxt(new Context);
        ctxt->provider = pva::getChannelProviderRegistry()->getProvider(pname);

        SELF.swap(ctxt);

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

        if(!SELF->provider)
            return PyErr_Format(PyExc_RuntimeError, "Context has been closed");

        Channel::shared_pointer req(new Channel);

        pva::Channel::shared_pointer chan;

        std::string chanName(cname);

        Context::channels_t::const_iterator it = SELF->channels.find(chanName);

        if(it!=SELF->channels.end()) {
            chan = it->second;

        } else {
            {
                PyUnlock U;
                chan = SELF->provider->createChannel(chanName, req);
            }
            //TODO: worry about concurrent connection to same channel?
            if(chan)
                SELF->channels[chanName] = chan;
        }

        if(!chan)
            return PyErr_Format(PyExc_RuntimeError, "Failed to create channel '%s'", cname);

        req->context = SELF;
        req->channel = chan;

        PyRef ret(PyChannel::type.tp_new(&PyChannel::type, args, kws));

        PyChannel::unwrap(ret.get()).swap(req);

        return ret.release();
    } CATCH()
    return NULL;
}

PyObject *Context::py_close(PyObject *self)
{
    TRY {
        if(SELF->provider) {
            SELF->provider.reset();
            Context::channels_t chans;
            chans.swap(SELF->channels);
            {
                PyUnlock U;
                chans.clear(); // may result in I/O
            }
        }
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
            PyRef name(PyString_FromString((*names)[i].c_str()));

            PyList_SET_ITEM(ret.get(), i, name.get());
        }

        return ret.release();
    }CATCH()
    return NULL;
}

pvd::PVStructure::shared_pointer buildRequest(PyObject *req)
{
    pvd::PVStructure::shared_pointer opts;

    /*TODO:
     *  None -> empty struct
     *  str  -> parse from pvRequest min-language
     *  {}   -> Translate directly
     *
     */
    if(req==Py_None) {
        // create an empty struct... argh!!!!
        opts = pvd::getPVDataCreate()->createPVStructure(pvd::getFieldCreate()->createFieldBuilder()->createStructure());

    } else if(PyString_Check(req)) {
        throw std::runtime_error("pvRequest parsing not implemented");

    } else {
        opts = P4PValue_unwrap(req);
    }
    return opts;
}

#undef TRY
#define TRY PyChannel::reference_type SELF = PyChannel::unwrap(self); try

PyObject* Channel::py_get(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"callback", "request"};
        PyObject *cb, *req = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O", (char**)names, &cb, &req))
            return NULL;

        if(!PyCallable_Check(cb))
            return PyErr_Format(PyExc_ValueError, "callable required, not %s", Py_TYPE(cb)->tp_name);

        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        GetOp::shared_pointer reqop(new GetOp(SELF));
        reqop->cb.reset(cb, borrow());
        reqop->req = buildRequest(req);

        SELF->ops.insert(reqop);

        reqop->restart(reqop);

        PyRef ret(PyOp::type.tp_new(&PyOp::type, args, kws));

        PyOp::unwrap(ret.get()) = reqop;

        return ret.release();
    }CATCH()
    return NULL;
}

void Channel::channelCreated(const pvd::Status& status, pva::Channel::shared_pointer const & channel)
{
    //TODO: can/do client contexts signal any errors here?
    if(!status.isOK()) {
        std::cout<<"Warning: unexpected in "<<__FUNCTION__<<" "<<status<<"\n";
    }
    (void)channel;
}

void Channel::channelStateChange(pva::Channel::shared_pointer const & channel, pva::Channel::ConnectionState connectionState)
{
    PyLock L;
    switch(connectionState) {
    case pva::Channel::NEVER_CONNECTED:
        break; // should never happen
    case pva::Channel::CONNECTED:
    {
        operations_t temp;
        temp.swap(ops);

        for(operations_t::const_iterator it = temp.begin(), end = temp.end(); it!=end; ++it) {
            if(!(*it)) continue; // shouldn't happen, but guard against it anyway
            try {
                (*it)->restart(*it);
                // restart() should re-add itself to ops
            } catch(std::exception& e) {
                std::cout<<"Error in restart() "<<e.what()<<"\n";
            }
        }
    }
        break;
    case pva::Channel::DISCONNECTED:
    {
        operations_t temp;
        temp.swap(ops);
        for(operations_t::const_iterator it = temp.begin(), end = temp.end(); it!=end; ++it) {
            if(!(*it)) continue; // shouldn't happen, but guard against it anyway
            try {
                (*it)->lostConn(*it);
            } catch(std::exception& e) {
                std::cout<<"Error in cancel() "<<e.what()<<"\n";
            }
        }
    }
        break;
    case pva::Channel::DESTROYED:
    {
        operations_t temp;
        temp.swap(ops);
        for(operations_t::const_iterator it = temp.begin(), end = temp.end(); it!=end; ++it) {
            if(!(*it)) continue; // shouldn't happen, but guard against it anyway
            try {
                (*it)->cancel();
            } catch(std::exception& e) {
                std::cout<<"Error in cancel() "<<e.what()<<"\n";
            }
        }
    }
        break;
    }
}


#undef TRY
#define TRY PyOp::reference_type SELF = PyOp::unwrap(self); try

PyObject* OpBase::py_cancel(PyObject *self)
{
    TRY {
        bool canceled = SELF->channel.get() && SELF->cancel();
        SELF->channel.reset();
        if(canceled)
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;

    } CATCH()
    return NULL;
}





void GetOp::restart(const GetOp::shared_pointer& self)
{
    pva::ChannelGet::shared_pointer temp;
    {
        PyUnlock U;
        try {
            temp = channel->channel->createChannelGet(self, req);
        } catch(std::runtime_error& e) {
            // TODO: what will happen if channel not connected?
            std::cout<<"Warning "<<__PRETTY_FUNCTION__<<" : "<<e.what()<<"\n";
        } catch(std::exception& e) {
            std::cout<<"Error "<<__PRETTY_FUNCTION__<<" : "<<e.what()<<"\n";
            return; // orphan
        }
    }
    op = temp;
    channel->ops.insert(self);
}

void GetOp::lostConn(const shared_pointer &self)
{
    channel->ops.insert(self);
    if(op) {
        pva::ChannelGet::shared_pointer temp;
        temp.swap(op);

        PyUnlock U;

        temp->destroy();
        temp.reset();
    }
}

bool GetOp::cancel()
{
    bool canceled = cb.get();

    if(op) {
        pva::ChannelGet::shared_pointer temp;
        temp.swap(op);

        PyUnlock U;

        temp->destroy();
        temp.reset();
    }

    return canceled;
}

void GetOp::channelGetConnect(
    const pvd::Status& status,
    pva::ChannelGet::shared_pointer const & channelGet,
    pvd::Structure::const_shared_pointer const & structure)
{
    // assume createChannelGet() return non-NULL
    if(!status.isOK()) {
        std::cerr<<__FUNCTION__<<" oops "<<status<<"\n";
    }
}

void GetOp::getDone(
    const pvd::Status& status,
    pva::ChannelGet::shared_pointer const & channelGet,
    pvd::PVStructure::shared_pointer const & pvStructure,
    pvd::BitSet::shared_pointer const & bitSet)
{
    PyLock L;
    if(!cb.get()) return;
    PyRef V;

    if(status.isSuccess()) {
        // we don't re-use ChannelGet, so assume exclusive ownership of pvStructure w/o a copy
        V.reset(P4PValue_wrap(P4PValue_type, pvStructure));
    } else {
        // build Exception instance
        // TODO: create RemoteError type
        V.reset(PyObject_CallFunction(PyExc_RuntimeError, (char*)"s", status.getMessage().c_str()));
    }

    if(V.get())
        V.reset(PyObject_CallFunctionObjArgs(cb.get(), V.get(), NULL));

    if(!V.get()) {
        PyErr_Print();
        PyErr_Clear();
    }
}


static PyMethodDef Context_methods[] = {
    {"channel", (PyCFunction)&Context::py_channel, METH_VARARGS|METH_KEYWORDS,
     "Return a Channel"},
    {"channel", (PyCFunction)&Context::py_close, METH_NOARGS,
     "Close this Context"},
    {"providers", (PyCFunction)&Context::py_providers, METH_NOARGS|METH_STATIC,
     "Return a list of all currently registered provider names"},
    {NULL}
};

template<>
PyTypeObject PyContext::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Context",
    sizeof(PyContext),
};

static PyMethodDef Channel_methods[] = {
    {"get", (PyCFunction)&Channel::py_get, METH_VARARGS|METH_KEYWORDS,
     "get(callback, request=None)\n\nInitiate a new get() operation.\n"
     "The provided callback must be a callable object, which will be called with a single argument.\n"
     "Either a Value or an Exception."},
    {NULL}
};

template<>
PyTypeObject PyChannel::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Channel",
    sizeof(PyChannel),
};

static PyMethodDef OpBase_methods[] = {
    {"cancel", (PyCFunction)&OpBase::py_cancel, METH_NOARGS,
     "Cancel pending operation."},
    {NULL}
};

template<>
PyTypeObject PyOp::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Operation",
    sizeof(PyOp),
};

} // namespace

void p4p_client_register(PyObject *mod)
{
    PyContext::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyContext::type.tp_new = &PyContext::tp_new;
    PyContext::type.tp_init = &Context::py_init;
    PyContext::type.tp_dealloc = &PyContext::tp_dealloc;

    PyContext::type.tp_methods = Context_methods;

    if(PyType_Ready(&PyContext::type))
        throw std::runtime_error("failed to initialize PyContext");

    Py_INCREF((PyObject*)&PyContext::type);
    if(PyModule_AddObject(mod, "Context", (PyObject*)&PyContext::type)) {
        Py_DECREF((PyObject*)&PyContext::type);
        throw std::runtime_error("failed to add p4p._p4p.Context");
    }


    PyChannel::type.tp_flags = Py_TPFLAGS_DEFAULT;
    PyChannel::type.tp_new = &PyChannel::tp_new;
    PyChannel::type.tp_init = &Context::py_init;
    PyChannel::type.tp_dealloc = &PyChannel::tp_dealloc;

    PyChannel::type.tp_methods = Channel_methods;

    if(PyType_Ready(&PyChannel::type))
        throw std::runtime_error("failed to initialize PyChannel");

    Py_INCREF((PyObject*)&PyChannel::type);
    if(PyModule_AddObject(mod, "Channel", (PyObject*)&PyChannel::type)) {
        Py_DECREF((PyObject*)&PyChannel::type);
        throw std::runtime_error("failed to add p4p._p4p.Channel");
    }


    PyOp::type.tp_flags = Py_TPFLAGS_DEFAULT;
    PyOp::type.tp_new = &PyOp::tp_new;
    PyOp::type.tp_init = &Context::py_init;
    PyOp::type.tp_dealloc = &PyOp::tp_dealloc;

    PyOp::type.tp_methods = OpBase_methods;

    if(PyType_Ready(&PyOp::type))
        throw std::runtime_error("failed to initialize PyOp");

    Py_INCREF((PyObject*)&PyOp::type);
    if(PyModule_AddObject(mod, "Operation", (PyObject*)&PyOp::type)) {
        Py_DECREF((PyObject*)&PyOp::type);
        throw std::runtime_error("failed to add p4p._p4p.Operation");
    }
}
