
#include <sstream>

#include <pv/configuration.h>
#include <pv/logger.h>
#include <pv/reftrack.h>
#include <pva/client.h>
#include "p4p.h"

namespace pvd = epics::pvData;
namespace pva = epics::pvAccess;

typedef PyClassWrapper<pvac::ClientProvider, true> PyClientProvider;
typedef PyClassWrapper<pvac::ClientChannel, true> PyClientChannel;

struct ClientMonitor : public pvac::ClientChannel::MonitorCallback {
    static size_t num_instances;

    epicsMutex pollLock;

    pvac::Monitor monitor;
    PyRef cb;

    ClientMonitor() {
        REFTRACE_INCREMENT(num_instances);
    }

    virtual ~ClientMonitor() {
        {
            PyUnlock U;
            monitor.cancel(); // we should be the only reference, but ... paranoia
        }
        REFTRACE_DECREMENT(num_instances);
    }

    virtual void monitorEvent(const pvac::MonitorEvent& evt)
    {
        PyLock L;
        TRACE(evt.event<<" "<<evt.message<<" -> "<<cb.get());

        if(!cb) return;

        PyRef ret(PyObject_CallFunction(cb.get(), "is", int(evt.event), evt.message.c_str()), allownull());
        if(!ret) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
        }
    }
};

size_t ClientMonitor::num_instances;

typedef PyClassWrapper<ClientMonitor> PyClientMonitor;

struct ClientOperation : public pvac::ClientChannel::PutCallback,
                         public pvac::ClientChannel::GetCallback
{
    static size_t num_instances;

    pvac::ClientChannel chan;
    pvac::Operation op;
    pvd::PVStructure::const_shared_pointer pvRequest;
    PyRef cb; // done callback
    PyRef builder;
    PyRef getval; // only for put

    ClientOperation() {
        REFTRACE_INCREMENT(num_instances);
    }
    virtual ~ClientOperation() {
        {
            PyUnlock U;
            op.cancel();
        }
        REFTRACE_DECREMENT(num_instances);
    }

    void prepvalue(PyRef& pyvalue, const pvd::PVStructure::const_shared_pointer& value, const pvd::BitSet* mask)
    {
        if(value) {
            assert(mask);
            pvd::PVStructure::shared_pointer V(pvd::getPVDataCreate()->createPVStructure(value->getStructure()));
            V->copyUnchecked(*value);
            pvd::BitSet::shared_pointer valid(new pvd::BitSet(*mask));

            pyvalue.reset(P4PValue_wrap(P4PValue_type, V, valid));
        } else {
            pyvalue.reset(Py_None, borrow());
        }
    }

    virtual void getDone(const pvac::GetEvent& evt)
    {
        PyLock L;
        TRACE(evt.event<<" '"<<evt.message<<"' "<<!!evt.value<<" -> "<<cb.get());

        if(!cb) return;

        PyRef pyvalue;
        prepvalue(pyvalue, evt.value, evt.valid.get());
        assert(pyvalue);

        PyRef ret(PyObject_CallFunction(cb.get(), "isO", int(evt.event), evt.message.c_str(), pyvalue.get()), allownull());
        if(!ret) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
        }
   }

    virtual void putBuild(const pvd::StructureConstPtr& build,
                          pvac::ClientChannel::PutCallback::Args& args)
    {
        PyLock L;

        PyRef pyvalue;
        prepvalue(pyvalue, args.previous, &args.previousmask);
        TRACE(pyvalue.get());

        if(!pyvalue) {
            // we didn't do a get first, so initalize
            pvd::PVStructure::shared_pointer value(pvd::getPVDataCreate()->createPVStructure(build));
            pvd::BitSet::shared_pointer valid(new pvd::BitSet);

            pyvalue.reset(P4PValue_wrap(P4PValue_type, value, valid));
        } else {
            // we did complete a get first, clear the valid mask
            // this is safe as we haven't (yet) pass this Value to a callback

            P4PValue_unwrap_bitset(pyvalue.get())->clear();
        }
        // builder callback is expected to populate the valid mask

        PyRef ret(PyObject_CallFunction(builder.get(), "O", pyvalue.get()), allownull());

        if(!ret) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
            // TODO: throwing here means that putDone() will be called with an Error.
            //       should we bother to save/restore the exception state instead of just printing?
            throw std::runtime_error("PyErr during builder callback");
        } else if(Py_REFCNT(pyvalue.get())!=1) {
            throw std::runtime_error("put builders must be synchronous and can not save the input value");
        }

        args.root = P4PValue_unwrap(pyvalue.get(), &args.tosend);
    }

    virtual void putDone(const pvac::PutEvent& evt)
    {
        PyLock L;
        TRACE(evt.event<<" '"<<evt.message<<"' -> "<<cb.get());

        if(!cb) return;

        PyRef ret(PyObject_CallFunction(cb.get(), "isO", int(evt.event), evt.message.c_str(), Py_None), allownull());
        if(!ret) {
            TRACE("ERROR");
            PyErr_Print();
            PyErr_Clear();
        }
    }
};

size_t ClientOperation::num_instances;

typedef PyClassWrapper<ClientOperation> PyClientOperation;

PyClassWrapper_DEF(PyClientProvider, "ClientProvider")
PyClassWrapper_DEF(PyClientChannel, "ClientChannel")
PyClassWrapper_DEF(PyClientMonitor, "ClientMonitor")
PyClassWrapper_DEF(PyClientOperation, "ClientOperation")

namespace {

#define TRY PyClientProvider::reference_type SELF = PyClientProvider::unwrap(self); try

static int clientprovider_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"provider", "conf", "useenv", NULL};
        const char *pname;
        PyObject *cdict = Py_None, *useenv = Py_True;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "s|OO", (char**)names, &pname, &cdict, &useenv))
            return -1;

        pva::ConfigurationBuilder B;

        if(PyObject_IsTrue(useenv)) {
            TRACE("useenv=true");
            B.push_env();
        }

        if(cdict==Py_None) {
            TRACE("No config");
            // nothing
        } else if(PyDict_Check(cdict)) {
            Py_ssize_t I = 0;
            PyObject *key, *value;

            while(PyDict_Next(cdict, &I, &key, &value)) {
                PyString K(key), V(value);

                B.add(K.str(), V.str());
                TRACE("config "<<K.str()<<"="<<V.str());
            }

            B.push_map();
        } else {
            PyErr_Format(PyExc_ValueError, "conf=%s not valid", Py_TYPE(cdict)->tp_name);
            return -1;
        }

        TRACE("");
        SELF = pvac::ClientProvider(pname, B.build());

        return 0;
    }CATCH()
    return -1;
}

static PyObject *clientprovider_close(PyObject *self)
{
    TRY {
        TRACE("");
        {
            PyUnlock U;
            SELF.reset();
        }
        Py_RETURN_NONE;
    }CATCH()
    return 0;
}

static PyObject* clientprovider_disconnect(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"name", NULL};
        const char *pchannel = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|z", (char**)names, &pchannel))
            return NULL;

        TRACE(pchannel);

        {
            PyUnlock U;
            if(pchannel)
                SELF.disconnect(pchannel);
            else
                SELF.disconnect();
        }

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject*  clientprovider_providers(PyObject *junk)
{
    try {
        pva::ChannelProviderRegistry::provider_name_set names;
        pva::ChannelProviderRegistry::clients()->getProviderNames(names);

        PyRef ret(PyList_New(names.size()));

        size_t i=0;
        for(pva::ChannelProviderRegistry::provider_name_set::const_iterator it = names.begin();
            it != names.end(); ++it) {
            PyRef name(PyUnicode_FromString(it->c_str()));

            PyList_SET_ITEM(ret.get(), i++, name.release());
        }

        return ret.release();
    }CATCH()
    return NULL;
}

PyObject*  clientprovider_set_debug(PyObject *junk, PyObject *args, PyObject *kws)
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

PyObject* clientprovider_makeRequest(PyObject *junk, PyObject *args)
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

static PyMethodDef clientprovider_methods[] = {
    {"close", (PyCFunction)&clientprovider_close, METH_NOARGS,
     "close()\n"
     "Shutdown provider and close all channels."},
    {"disconnect", (PyCFunction)&clientprovider_disconnect, METH_VARARGS|METH_KEYWORDS,
     "disconnect(channel=None)\n"
     "Clear internal channel cache.  Allows unused ClientChannels to implicitly close."},
    {"providers", (PyCFunction)&clientprovider_providers, METH_NOARGS|METH_STATIC,
     "providers() -> ['name', ...]\n"
     ":returns: A list of all currently registered provider names.\n\n"
     "A staticmethod."},
    {"set_debug", (PyCFunction)&clientprovider_set_debug, METH_VARARGS|METH_KEYWORDS|METH_STATIC,
     "set_debug(lvl)\n\n"
     "Set PVA debug level"},
    {"makeRequest", (PyCFunction)&clientprovider_makeRequest, METH_VARARGS|METH_STATIC,
     "makeRequest(\"field(value)\")\n\nParse pvRequest string"},
    {NULL}
};

#undef TRY
#define TRY PyClientChannel::reference_type SELF = PyClientChannel::unwrap(self); try

static int clientchannel_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"ctxt", "name", "address", "priority", NULL};
        PyObject *pyprovider;
        const char *name, *address = 0;
        short prio = 0;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O!s|zh", (char**)names,
                                        &PyClientProvider::type, &pyprovider,
                                        &name, &address, &prio))
            return -1;

        pvac::ClientChannel::Options opts;
        opts.priority = prio;
        if(address)
            opts.address = address;

        pvac::ClientProvider prov(PyClientProvider::unwrap(pyprovider));

        {
            PyUnlock U;
            SELF = prov.connect(name, opts);
        }

        TRACE(name);
        return 0;
    }CATCH()
    return -1;
}

static PyObject* clientchannel_show(PyObject *self)
{
    TRY {
        std::ostringstream strm;

        {
            PyUnlock U;
            SELF.show(strm);
        }

        return PyUnicode_FromString(strm.str().c_str());
    }CATCH()
    return 0;
}

static PyMethodDef clientchannel_methods[] = {
    {"show", (PyCFunction)&clientchannel_show, METH_NOARGS,
     "show() -> str\n"
     "String representation"},
    {NULL}
};

#undef TRY
#define TRY PyClientMonitor::reference_type SELF = PyClientMonitor::unwrap(self); try

static int clientmonitor_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"channel", "handler", "pvRequest", NULL};
        PyObject *chan, *cb, *pvReq = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O!O|O", (char**)names,
                                        &PyClientChannel::type, &chan, &cb, &pvReq))
            return -1;

        pvd::PVStructure::const_shared_pointer pvRequest;

        if(pvReq==Py_None) {
        } else {
            pvRequest = P4PValue_unwrap(pvReq);
        }

        pvac::ClientChannel& channel = PyClientChannel::unwrap(chan);

        SELF.cb.reset(cb, borrow());
        {
            PyUnlock U;
            SELF.monitor = channel.monitor(&SELF, pvRequest);
        }
        TRACE(channel.name());

        return 0;
    }CATCH()
    return -1;
}

static PyObject *clientmonitor_close(PyObject *self)
{
    TRY {
        TRACE("");
        {
            PyUnlock U;
            Guard G(SELF.pollLock);
            SELF.monitor.cancel();
        }
        Py_RETURN_NONE;
    }CATCH()
    return 0;
}

static PyObject *clientmonitor_pop(PyObject *self)
{
    TRY {
        pvd::PVStructure::shared_pointer root;
        pvd::BitSet::shared_pointer changed;
        {
            PyUnlock U;
            // until https://github.com/epics-base/pvAccessCPP/commit/24e83daaba1c7b3618b355b4f4dc37ab79414a74
            // cancel() also cleared monitor.root et al.
            // use pollLock to avoid race
            Guard G(SELF.pollLock);

            if(SELF.monitor.poll()) {
                assert(SELF.monitor.root.get());
                root = pvd::getPVDataCreate()->createPVStructure(SELF.monitor.root->getStructure());
                root->copyUnchecked(*SELF.monitor.root);
                changed.reset(new pvd::BitSet(SELF.monitor.changed));
                // TODO: something with SELF.monitor.overrun
            }
        }
        if(root) {
            TRACE("Value");
            return P4PValue_wrap(P4PValue_type, root, changed);
        } else {
            TRACE("None");
            Py_RETURN_NONE;
        }
    }CATCH()
    return 0;
}

static PyObject *clientmonitor_complete(PyObject *self)
{
    TRY {
        bool ret;
        {
            PyUnlock U;
            ret = SELF.monitor.complete();
        }
        if(ret) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    }CATCH()
    return 0;
}

static PyMethodDef clientmonitor_methods[] = {
    {"close", (PyCFunction)&clientmonitor_close, METH_NOARGS,
     "close()\n"
     "Cancel subscription"},
    {"pop", (PyCFunction)&clientmonitor_pop, METH_NOARGS,
     "pop() -> Value | None\n"
     "Pop next element from subscription FIFO"},
    {"complete", (PyCFunction)&clientmonitor_complete, METH_NOARGS,
     "complete() -> bool\n"
     "Has this subscription seen its final update.  Call after poll()."},
    {NULL}
};

static int clientmonitor_traverse(PyObject *self, visitproc visit, void *arg)
{
    TRY {
        if(SELF.cb) {
            Py_VISIT(SELF.cb.get());
        }
        return 0;
    } CATCH()
    return -1;
}

static int clientmonitor_clear(PyObject *self)
{
    TRY {
        if(SELF.cb) {
            PyRef tmp;
            SELF.cb.swap(tmp);
        }
        return 0;
    } CATCH()
    return -1;
}

#undef TRY
#define TRY PyClientOperation::reference_type SELF = PyClientOperation::unwrap(self); try

static int clientoperation_init(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        static const char* names[] = {"channel", "handler",
                                      "value", "builder", "pvRequest", "get", "put", "rpc", NULL};
        PyObject *chan, *cb;
        PyObject *pyvalue = Py_None, *builder = Py_None, *pvReq = Py_None,
                 *doGet = Py_False, *doPut = Py_False, *doRPC = Py_False;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O!O|OOOOOO", (char**)names,
                                        &PyClientChannel::type, &chan, &cb,
                                        &pyvalue, &builder, &pvReq, &doGet, &doPut, &doRPC))
            return -1;

        pvd::PVStructure::const_shared_pointer pvRequest;

        if(pvReq!=Py_None) {
            pvRequest = P4PValue_unwrap(pvReq);
        }

        pvac::ClientChannel& channel = PyClientChannel::unwrap(chan);

        SELF.cb.reset(cb, borrow());
        SELF.pvRequest = pvRequest;
        SELF.chan = channel;

        bool get = PyObject_IsTrue(doGet),
             put = PyObject_IsTrue(doPut),
             rpc = PyObject_IsTrue(doRPC);
        TRACE(get<<" "<<put<<" "<<rpc);

        if(put && !rpc) {
            TRACE("put"<<(get?" w/ get":""));
            if(!PyCallable_Check(builder)) {
                PyErr_Format(PyExc_ValueError, "Operation put=True requires builder= callable");
                return -1;
            }
            SELF.builder.reset(builder, borrow());
            PyUnlock U;
            SELF.op = channel.put(&SELF, pvRequest, get);
        } else if(get && !put && !rpc) {
            // only get
            TRACE("only get");
            PyUnlock U;
            SELF.op = channel.get(&SELF, pvRequest);
        } else if(!get && !put && rpc) {
            TRACE("rpc");
            pvd::PVStructure::const_shared_pointer value(P4PValue_unwrap(pyvalue));
            PyUnlock U;
            SELF.op = channel.rpc(&SELF, value, pvRequest);
        } else {
            PyErr_Format(PyExc_ValueError, "Operation unsupported combination of get=, put=, and rpc=");
            return -1;
        }

        TRACE("");
        return 0;
    }CATCH()
    return -1;
}

static PyObject *clientoperation_close(PyObject *self)
{
    TRY {
        TRACE("");
        {
            PyUnlock U;
            SELF.op.cancel();
        }
        Py_RETURN_NONE;
    }CATCH()
    return 0;
}

static PyMethodDef clientoperation_methods[] = {
    {"close", (PyCFunction)&clientoperation_close, METH_NOARGS,
     "close()\n"
     "Cancel pending operation"},
    {NULL}
};

static int clientoperation_traverse(PyObject *self, visitproc visit, void *arg)
{
    TRY {
        if(SELF.cb) {
            Py_VISIT(SELF.cb.get());
        }
        if(SELF.builder) {
            Py_VISIT(SELF.builder.get());
        }
        if(SELF.getval) {
            Py_VISIT(SELF.getval.get());
        }
        return 0;
    } CATCH()
    return -1;
}

static int clientoperation_clear(PyObject *self)
{
    TRY {
        if(SELF.cb) {
            PyRef tmp;
            SELF.cb.swap(tmp);
        }
        if(SELF.builder) {
            PyRef tmp;
            SELF.builder.swap(tmp);
        }
        if(SELF.getval) {
            PyRef tmp;
            SELF.getval.swap(tmp);
        }
        return 0;
    } CATCH()
    return -1;
}

#undef TRY

} //namespace

void p4p_client_register(PyObject *mod)
{
    epics::registerRefCounter("p4p._p4p.ClientMonitor", &ClientMonitor::num_instances);
    epics::registerRefCounter("p4p._p4p.ClientOperation", &ClientOperation::num_instances);

    PyClientProvider::buildType();

    PyClientProvider::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyClientProvider::type.tp_init = &clientprovider_init;

    PyClientProvider::type.tp_methods = clientprovider_methods;

    PyClientProvider::finishType(mod, "ClientProvider");


    PyClientChannel::buildType();

    PyClientChannel::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    PyClientChannel::type.tp_init = &clientchannel_init;

    PyClientChannel::type.tp_methods = clientchannel_methods;

    PyClientChannel::finishType(mod, "ClientChannel");


    PyClientMonitor::buildType();

    PyClientMonitor::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC;
    PyClientMonitor::type.tp_init = &clientmonitor_init;
    PyClientMonitor::type.tp_traverse = &clientmonitor_traverse;
    PyClientMonitor::type.tp_clear = &clientmonitor_clear;

    PyClientMonitor::type.tp_methods = clientmonitor_methods;

    PyClientMonitor::finishType(mod, "ClientMonitor");


    PyClientOperation::buildType();

    PyClientOperation::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC;
    PyClientOperation::type.tp_init = &clientoperation_init;
    PyClientOperation::type.tp_traverse = &clientoperation_traverse;
    PyClientOperation::type.tp_clear = &clientoperation_clear;

    PyClientOperation::type.tp_methods = clientoperation_methods;

    PyClientOperation::finishType(mod, "ClientOperation");
}
