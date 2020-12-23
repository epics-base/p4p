
#include <sstream>

#include "p4p.h"
#include <_p4p.h>

namespace p4p {

std::string toString(const server::Server& serv, int detail)
{
    std::ostringstream strm;
    Detailed D(strm, detail);
    strm<<serv;
    return strm.str();
}

void attachHandler(server::SharedPV& pv, PyObject *handler)
{
    // we assume the caller will maintain a ref. to handler until detachHandler() returns

    {
        PyUnlock U;

        pv.onFirstConnect([handler](server::SharedPV&){
            PyLock L;

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "onFirstConnect", "")));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
                // continue to build the channel
            }
        });

        pv.onLastDisconnect([handler](server::SharedPV&){
            PyLock L;

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "onLastDisconnect", "")));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                // we don't treat this failure as critical.
                // continue to build the channel
            }
        });

        pv.onPut([handler](server::SharedPV&, std::unique_ptr<server::ExecOp>&& rawop, Value&& val){
            PyLock L;
            std::shared_ptr<server::ExecOp> op(std::move(rawop));
            PyRef pyop(ServerOperation_wrap(op, val));

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "put", "O", pyop.obj)));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                op->error("Internal Error on Remote end");
            }
        });

        pv.onRPC([handler](server::SharedPV&, std::unique_ptr<server::ExecOp>&& rawop, Value&& val){
            PyLock L;
            std::shared_ptr<server::ExecOp> op(std::move(rawop));
            PyRef pyop(ServerOperation_wrap(op, val));

            auto ret(PyRef::allownull(PyObject_CallMethod(handler, "rpc", "O", pyop.obj)));
            if(PyErr_Occurred()) {
                PySys_WriteStderr("Unhandled Exception %s:%d\n", __FILE__, __LINE__);
                PyErr_Print();
                PyErr_Clear();
                op->error("Internal Error on Remote end");
            }
        });
    }
}

void detachHandler(server::SharedPV& pv)
{
    PyUnlock U;

    pv.onFirstConnect(nullptr);

    pv.onLastDisconnect(nullptr);

    pv.onPut(nullptr);

    pv.onRPC(nullptr);
}

}
