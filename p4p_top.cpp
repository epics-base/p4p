
#include "p4p.h"

//TODO: drop support for numpy 1.6 (found in debian <=7)
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
#include <numpy/ndarrayobject.h>

PyMOD(_p4p)
{
    try {
#if PY_MAJOR_VERSION >= 3
        PyRef mod(PyModule_Create(&inotifymodule));
#else
        PyRef mod(Py_InitModule("_p4p", NULL));
#endif

        import_array();

        p4p_type_register(mod.get());
        p4p_value_register(mod.get());
        p4p_server_register(mod.get());

        MODINIT_RET(mod.release());
    } catch(std::exception& e) {
        PySys_WriteStderr("Import of _p4p failed: %s\n", e.what());
        MODINIT_RET(NULL);
    }
}
