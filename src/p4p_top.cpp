
#include "p4p.h"

//TODO: drop support for numpy 1.6 (found in debian <=7)
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
#include <numpy/ndarrayobject.h>

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef p4pymodule = {
  PyModuleDef_HEAD_INIT,
    "_p4p",
    NULL,
    -1,
    NULL
};
#endif

PyMOD(_p4p)
{
    try {
#if PY_MAJOR_VERSION >= 3
        PyRef mod(PyModule_Create(&p4pymodule));
#else
        PyRef mod(Py_InitModule("_p4p", P4P_methods));
#endif

        import_array();

        p4p_type_register(mod.get());
        p4p_value_register(mod.get());
        p4p_array_register(mod.get());
        p4p_server_register(mod.get());
        p4p_server_provider_register(mod.get());
        p4p_client_register(mod.get());

        MODINIT_RET(mod.release());
    } catch(std::exception& e) {
        PySys_WriteStderr("Import of _p4p failed: %s\n", e.what());
        MODINIT_RET(NULL);
    }
}

PyRef::PyRef(const PyExternalRef& o) :obj(o.ref.obj) {
    Py_XINCREF(obj);
}
