
#include "p4p.h"
#include <_p4p.h>

namespace p4p {

namespace {

// cf. pvac::PutEvent::event_t
enum struct GPRResult {
    Fail,    //!< request ends in failure.  Check message
    Cancel,  //!< request cancelled before completion
    Success, //!< It worked!
};

} // namespace

// caller responsible for maintaining handler ref through callback
std::function<void(client::Result&&)> opHandler(PyObject *handler)
{
    return [handler](client::Result&& result) mutable {
        GPRResult event;
        std::string msg;
        Value val;
        try {
            val = result();
            event = GPRResult::Success;

        }catch(std::exception& err){
            event = GPRResult::Fail;
            msg = err.what();
        }

        PyLock L;

        PyRef pyval;
        PyObject *arg = Py_None;
        if(val) {
            pyval.reset(pvxs_pack(val));
            arg = pyval.obj;
        }

        auto ret(PyRef::allownull(PyObject_CallFunction(handler, "isO", (int)event, msg.c_str(), arg)));
        if(!ret.obj) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();
        }
    };
}

// caller responsible for maintaining handler ref
std::function<Value (Value &&)> opBuilder(PyObject *handler)
{
    return [handler](Value&& raw) mutable -> Value {
        auto val(std::move(raw));

        // previous field values should not be re-sent marked
        val.unmark();

        PyLock L;

        PyRef arg(pvxs_pack(val));

        auto ret(PyRef::allownull(PyObject_CallFunction(handler, "O", arg.obj)));
        if(!ret.obj) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();
        } else if(Py_REFCNT(arg.obj)!=1) {
            throw std::logic_error("put builders must be synchronous and can not save the input value");
        }

        return val;
    };
}

// caller responsible for maintaining handler ref
void opEvent(client::MonitorBuilder& builder, PyObject *handler)
{
    builder.event([handler](client::Subscription&) mutable {

        PyLock L;

        auto ret(PyRef::allownull(PyObject_CallFunction(handler, "")));
        if(!ret.obj) {
            PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
            PyErr_Print();
            PyErr_Clear();
        }
    });
}

PyObject* monPop(const std::shared_ptr<client::Subscription>& mon)
{
    PyObject* klass;
    std::string msg;
    try {
        Value ret;
        {
            PyUnlock U;
            ret = mon->pop();
        }
        if(ret)
            return pvxs_pack(ret);
        else
            Py_RETURN_NONE;
    //}catch(client::Connected&){ // masked
    }catch(client::Finished& e){
        klass = _Finished;
        msg = e.what();
    }catch(client::Disconnect& e){
        klass = _Disconnected;
        msg = e.what();
    }catch(client::RemoteError& e){
        klass = _RemoteError;
        msg = e.what();
    }

    return PyObject_CallFunction(klass, "s", msg.c_str());
}
} // namespace p4p
