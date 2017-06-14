
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

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

Channel::~Channel()
{
    pva::Channel::shared_pointer chan;
    channel.swap(chan);
    if(chan) {
        PyUnlock U;
        chan.reset();
    }
}

void Channel::Req::channelCreated(const pvd::Status& status, pva::Channel::shared_pointer const & channel)
{
    //TODO: can/do client contexts signal any errors here?
    TRACE(channel->getChannelName()<<" "<<status);
    if(!status.isOK()) {
        std::cout<<"Warning: unexpected in "<<__FUNCTION__<<" "<<status<<"\n";
    }
    (void)channel;
}

void Channel::Req::channelStateChange(pva::Channel::shared_pointer const & channel, pva::Channel::ConnectionState connectionState)
{
    Channel::shared_pointer op(owner.lock());
    if(!op)
        return;
    TRACE(channel->getChannelName()<<" "<<connectionState);
}

namespace {

#define TRY PyChannel::reference_type SELF = PyChannel::unwrap(self); try

int py_channel_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        // ignores arguments
        if(!SELF->channel) {
            PyErr_Format(PyExc_ValueError, "Can't construct Channel explicitly");
            return 1;
        }

        return 0;
    }CATCH();
    return 1;
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
        // create {'field':{}} to get everything
        opts = pvd::getPVDataCreate()->createPVStructure(pvd::getFieldCreate()->createFieldBuilder()
                                                         ->addNestedStructure("field")
                                                         ->endNested()
                                                         ->createStructure());

    } else if(PyBytes_Check(req) || PyUnicode_Check(req)) {
        PyString S(req);
        std::string R(S.str());
        if(R.empty())
            R = "field()"; // ensure some sane default (get everything)

        pvd::CreateRequest::shared_pointer create(pvd::CreateRequest::create());
        opts = create->createRequest(R);
        if(!opts)
            throw std::runtime_error(SB()<<"Error parsing pvRequest \""<<R<<"\" : "<<create->getMessage());

    } else {
        opts = P4PValue_unwrap(req);
    }
    return opts;
}

PyObject *py_channel_get(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"callback", "request", NULL};
        PyObject *cb, *req = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O", (char**)names, &cb, &req))
            return NULL;

        if(!PyCallable_Check(cb))
            return PyErr_Format(PyExc_ValueError, "callable required, not %s", Py_TYPE(cb)->tp_name);

        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        TRACE("Channel get "<<SELF->channel->getChannelName());

        GetOp::shared_pointer reqop(new GetOp);
        GetOp::Req::shared_pointer pyreq(new GetOp::Req(reqop));
        pvd::PVStructure::shared_pointer pvReq(buildRequest(req));

        reqop->cb.reset(cb, borrow());

        GetOp::operation_t::shared_pointer op(SELF->channel->createChannelGet(pyreq, pvReq));
        if(!op) {
            Py_RETURN_NONE;
        } else if(!op.unique()) {
            std::cerr<<"Provider "<<SELF->channel->getProvider()->getProviderName()
                     <<" for "<<SELF->channel->getChannelName()
                     <<" gives non-unique Get operation.  use_count="<<op.use_count()<<"\n";
        }
        // cb may be called at this point.
        reqop->op.swap(op);

        try {
            PyRef ret(PyOp::type.tp_new(&PyOp::type, args, kws));

            PyOp::unwrap(ret.get()) = reqop;

            return ret.release();
        }catch(...) {
            reqop->op->destroy(); // paranoia?
            throw;
        }
    }CATCH()
    return NULL;
}

PyObject *py_channel_put(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"callback", "value", "request", NULL};
        PyObject *cb, *val, *req = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "OO|O", (char**)names, &cb, &val, &req))
            return NULL;

        if(!PyCallable_Check(cb))
            return PyErr_Format(PyExc_ValueError, "callable required, not %s", Py_TYPE(cb)->tp_name);

        if(PyObject_IsInstance(val, (PyObject*)P4PValue_type)) { /* ok */ }
        else if(PyCallable_Check(val)) { /* ok */ }
        else {
            return PyErr_Format(PyExc_ValueError, "put value must be Value or callable which returns Value");
        }

        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        TRACE("Channel put "<<SELF->channel->getChannelName());

        PutOp::shared_pointer reqop(new PutOp);
        PutOp::Req::shared_pointer pyreq(new PutOp::Req(reqop));
        pvd::PVStructure::shared_pointer pvReq(buildRequest(req));

        reqop->cb.reset(cb, borrow());
        reqop->pyvalue.reset(val, borrow());

        PutOp::operation_t::shared_pointer op(SELF->channel->createChannelPut(pyreq, pvReq));
        if(!op) {
            Py_RETURN_NONE;
        } else if(!op.unique()) {
            std::cerr<<"Provider "<<SELF->channel->getProvider()->getProviderName()
                     <<" for "<<SELF->channel->getChannelName()
                     <<" gives non-unique Put operation.  use_count="<<op.use_count()<<"\n";
        }
        // cb may be called at this point.
        reqop->op.swap(op);

        try {
            PyRef ret(PyOp::type.tp_new(&PyOp::type, args, kws));

            PyOp::unwrap(ret.get()) = reqop;

            return ret.release();
        }catch(...) {
            reqop->op->destroy();
            throw;
        }
    }CATCH()
    return NULL;
}

PyObject *py_channel_rpc(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"callback", "value", "request", NULL};
        PyObject *cb, *val, *req = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "OO!|O", (char**)names, &cb, P4PValue_type, &val, &req))
            return NULL;

        if(!PyCallable_Check(cb))
            return PyErr_Format(PyExc_ValueError, "callable required, not %s", Py_TYPE(cb)->tp_name);

        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        TRACE("Channel rpc "<<SELF->channel->getChannelName());

        RPCOp::shared_pointer reqop(new RPCOp);
        RPCOp::Req::shared_pointer pyreq(new RPCOp::Req(reqop));
        pvd::PVStructure::shared_pointer pvReq(buildRequest(req));

        reqop->cb.reset(cb, borrow());
        reqop->pvvalue = P4PValue_unwrap(val);

        RPCOp::operation_t::shared_pointer op;
        {
            PyUnlock U;
            op = SELF->channel->createChannelRPC(pyreq, pvReq);
        }

        if(!op) {
            Py_RETURN_NONE;
        } else if(!op.unique()) {
            std::cerr<<"Provider "<<SELF->channel->getProvider()->getProviderName()
                     <<" for "<<SELF->channel->getChannelName()
                     <<" gives non-unique RPC operation.  use_count="<<op.use_count()<<"\n";
        }
        // cb may be called at this point.
        reqop->op.swap(op);

        try {
            PyRef ret(PyOp::type.tp_new(&PyOp::type, args, kws));

            PyOp::unwrap(ret.get()) = reqop;

            return ret.release();
        }catch(...) {
            reqop->op->destroy();
            throw;
        }
    }CATCH()
    return NULL;
}

PyObject *py_channel_monitor(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char *names[] = {"callback", "request", NULL};
        PyObject *cb, *req = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O|O", (char**)names, &cb, &req))
            return NULL;

        if(!PyCallable_Check(cb))
            return PyErr_Format(PyExc_ValueError, "callable required, not %s", Py_TYPE(cb)->tp_name);

        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        TRACE("Channel monitor "<<SELF->channel->getChannelName()<<" cb="<<cb);

        MonitorOp::shared_pointer reqop(new MonitorOp);
        MonitorOp::Req::shared_pointer pyreq(new MonitorOp::Req(reqop));
        pvd::PVStructure::shared_pointer pvReq(buildRequest(req));

        reqop->event.reset(cb, borrow());

        MonitorOp::operation_t::shared_pointer op(SELF->channel->createMonitor(pyreq, pvReq));
        if(!op) {
            Py_RETURN_NONE;
        } else if(!op.unique()) {
            std::cerr<<"Provider "<<SELF->channel->getProvider()->getProviderName()
                     <<" for "<<SELF->channel->getChannelName()
                     <<" gives non-unique Monitor operation.  use_count="<<op.use_count()<<"\n";
        }
        // cb may be called at this point.
        reqop->op.swap(op);

        try {
            PyRef ret(PyOp::type.tp_new(&PyMonitorOp::type, args, kws));

            PyMonitorOp::unwrap(ret.get()) = reqop;

            return ret.release();
        }catch(...) {
            reqop->op->destroy();
            throw;
        }
    }CATCH()
    return NULL;
}

PyObject *py_channel_name(PyObject *self)
{
    TRY {
        if(!SELF->channel)
            return PyErr_Format(PyExc_RuntimeError, "Channel closed");

        return PyUnicode_FromString(SELF->channel->getChannelName().c_str());
    }CATCH();
    return NULL;
}

PyObject *py_channel_close(PyObject *self)
{
    TRY {
        if(SELF->channel) {
            pva::Channel::shared_pointer chan;
            SELF->channel.swap(chan);
            {
                PyUnlock U;
                chan->destroy();
                chan.reset();
            }
        }
        Py_RETURN_NONE;
    }CATCH();
    return NULL;
}

static PyMethodDef Channel_methods[] = {
    {"getName", (PyCFunction)&py_channel_name, METH_NOARGS,
     "Channel name (aka PV name)"},
    {"get", (PyCFunction)&py_channel_get, METH_VARARGS|METH_KEYWORDS,
     "get(callback, request=None)\n\nInitiate a new get() operation.\n"
     "The provided callback must be a callable object, which will be called with a single argument.\n"
     "Either a True (new data), False (no more data ever), None (Channel disconnected) or an Exception."},
    {"put", (PyCFunction)&py_channel_put, METH_VARARGS|METH_KEYWORDS,
     "put(callback, value, request=None)\n\nInitiate a new put() operation.\n"
     "The provided callback must be a callable object, which will be called with a single argument.\n"
     "Either None or an Exception."},
    {"rpc", (PyCFunction)&py_channel_rpc, METH_VARARGS|METH_KEYWORDS,
     "rpc(callback, value, request=None)\n\nInitiate a new rpc() operation.\n"
     "The provided callback must be a callable object, which will be called with a single argument.\n"
     "Either None or an Exception."},
    {"monitor", (PyCFunction)&py_channel_monitor, METH_VARARGS|METH_KEYWORDS,
     "monitor(callback, request=None)\n\nInitiate a new rpc() operation.\n"
     "The provided callback must be a callable object, which will be called with a single argument.\n"
     "Either None or an Exception."},
    {"close", (PyCFunction)&py_channel_close, METH_NOARGS,
      "close()\n\nDispose of channel."},
    {NULL}
};

} // namespace

template<>
PyTypeObject PyChannel::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "p4p._p4p.Channel",
    sizeof(PyChannel),
};

void p4p_client_channel_register(PyObject *mod)
{
    PyChannel::buildType();
    PyChannel::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyChannel::type.tp_init = &py_channel_init;

    PyChannel::type.tp_methods = Channel_methods;

    if(PyType_Ready(&PyChannel::type))
        throw std::runtime_error("failed to initialize PyChannel");

    Py_INCREF((PyObject*)&PyChannel::type);
    if(PyModule_AddObject(mod, "Channel", (PyObject*)&PyChannel::type)) {
        Py_DECREF((PyObject*)&PyChannel::type);
        throw std::runtime_error("failed to add p4p._p4p.Channel");
    }
}
