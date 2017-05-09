/* The P4PArray type exists only to act as the base object for a numpy array
 */
#include <stddef.h>

#include "p4p.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace {

namespace pvd = epics::pvData;

typedef PyClassWrapper<array_type > P4PArray;

template<>
PyTypeObject P4PArray::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_p4p.Array",
    sizeof(P4PArray),
};

} // namespace

PyTypeObject* P4PArray_type = &P4PArray::type;

PyObject* P4PArray_make(const array_type& v)
{
    PyRef ret(P4PArray::type.tp_new(&P4PArray::type, NULL, NULL));
    //TODO: call tp_init() ?
    P4PArray::unwrap(ret.get()) = v;
    return ret.release();
}

const array_type& P4PArray_extract(PyObject* o)
{
    if(Py_TYPE(o)!=&P4PArray::type)
        throw std::runtime_error(SB()<<"Can't extract vector from "<<Py_TYPE(o)->tp_name);
    return P4PArray::unwrap(o);
}

void p4p_array_register(PyObject *mod)
{
    P4PArray::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    P4PArray::type.tp_new = &P4PArray::tp_new;
    P4PArray::type.tp_dealloc = &P4PArray::tp_dealloc;

    //P4PArray::type.tp_weaklistoffset = offsetof()

    P4PArray::type.tp_doc = "Holder for a shared_array<> being shared w/ numpy";

    if(PyType_Ready(&P4PArray::type))
        throw std::runtime_error("failed to initialize P4PArray_type");

    Py_INCREF((PyObject*)&P4PArray::type);
    if(PyModule_AddObject(mod, "Array", (PyObject*)&P4PArray::type)) {
        Py_DECREF((PyObject*)&P4PArray::type);
        throw std::runtime_error("failed to add _p4p.Array");
    }
}
