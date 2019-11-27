
#include <time.h>

#include <pv/logger.h>

#include <pv/pvIntrospect.h> /* for pv/pvdVersion.h */
#include <pv/pvData.h>
#include <pv/pvAccess.h>
#include <pv/pvaVersion.h>
#include <pv/security.h>

#include "p4p.h"

// we only need to export the module init function
#define epicsExportSharedSymbols
#include <shareLib.h>

//TODO: drop support for numpy 1.6 (found in debian <=7)
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
#include <numpy/ndarrayobject.h>

PyObject* P4PCancelled;

PyObject* p4p_pvd_version(PyObject *junk)
{
#ifndef EPICS_PVD_MAJOR_VERSION
#define EPICS_PVD_MAJOR_VERSION 0
#define EPICS_PVD_MINOR_VERSION 0
#define EPICS_PVD_MAINTENANCE_VERSION 0
#define EPICS_PVD_DEVELOPMENT_FLAG 0
#endif
    return Py_BuildValue("iiii",
                         int(EPICS_PVD_MAJOR_VERSION),
                         int(EPICS_PVD_MINOR_VERSION),
                         int(EPICS_PVD_MAINTENANCE_VERSION),
                         int(EPICS_PVD_DEVELOPMENT_FLAG));
}

PyObject* p4p_pva_version(PyObject *junk)
{
    return Py_BuildValue("iiii",
                         int(EPICS_PVA_MAJOR_VERSION),
                         int(EPICS_PVA_MINOR_VERSION),
                         int(EPICS_PVA_MAINTENANCE_VERSION),
                         int(EPICS_PVA_DEVELOPMENT_FLAG));
}

PyObject* p4p_getrefs(PyObject *junk, PyObject *args, PyObject *kws)
{
    try {
        unsigned zeros = 0;
        const char *names[] = {"zeros", 0};
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|I", (char**)names, &zeros))
            return 0;

        epics::RefSnapshot snap;
        snap.update();

        PyRef ret(PyDict_New());

        for(epics::RefSnapshot::const_iterator it=snap.begin(), end=snap.end();
            it!=end; ++it)
        {
            if(!zeros && !it->second.current)
                continue;
            PyRef val(PyLong_FromSize_t(it->second.current));
            if(PyDict_SetItemString(ret.get(), it->first.c_str(), val.get()))
                throw std::runtime_error("");
        }

        return ret.release();
    }CATCH()
    return 0;
}

PyObject* p4p_force_lazy(PyObject *junk)
{
    try {
        (void)epics::pvData::getFieldCreate();
        (void)epics::pvData::getPVDataCreate();
        (void)epics::pvAccess::ChannelProviderRegistry::clients();
        (void)epics::pvAccess::AuthenticationRegistry::clients();

        Py_RETURN_NONE;
    }CATCH()
    return 0;
}

PyObject* p4p_serialize(PyObject *junk, PyObject *args, PyObject *kws)
{
    try {
        PyObject* obj;
        int BE = 0;
        const char *names[] = {"object", "be", 0};
        if(!PyArg_ParseTupleAndKeywords(args, kws, "O|p", (char**)names, &obj, &BE))
            return 0;

        std::tr1::shared_ptr<const epics::pvData::Serializable> fld;

        if(PyObject_IsInstance(obj, (PyObject*)P4PType_type)) {
            fld = P4PType_unwrap(obj);
        }

        if(!fld)
            return PyErr_Format(PyExc_ValueError, "Serialization of %s not supported", Py_TYPE(obj)->tp_name);

        std::vector<epicsUInt8> buf;
        epics::pvData::serializeToVector(fld.get(),
                                         BE ? EPICS_ENDIAN_BIG : EPICS_ENDIAN_LITTLE,
                                         buf);

        return PyBytes_FromStringAndSize((char*)&buf[0], buf.size());
    }CATCH()
    return 0;
}

static struct PyMethodDef P4P_methods[] = {
    {"installProvider", (PyCFunction)p4p_add_provider, METH_VARARGS|METH_KEYWORDS,
     "installProvider(\"name\", provider)\n"
     "Install a new Server Channel provider"},
    {"removeProvider", (PyCFunction)p4p_remove_provider, METH_VARARGS|METH_KEYWORDS,
     "removeProvider(\"name\")\n"
     "Remove a previously added Server Channel provider"},
    {"clearProviders", (PyCFunction)p4p_remove_all, METH_VARARGS|METH_KEYWORDS,
     "Remove all Server Channel providers"},
    {"pvdVersion", (PyCFunction)p4p_pvd_version, METH_NOARGS,
     ":returns: tuple of version number components for PVData"},
    {"pvaVersion", (PyCFunction)p4p_pva_version, METH_NOARGS,
     ":returns: tuple of version number components for PVData"},
    {"listRefs", (PyCFunction)p4p_getrefs, METH_VARARGS|METH_KEYWORDS,
     "listRefs(zeros=False)\n\n"
     "Snapshot c++ reference counter values.\n"
     "If zeros is False, then counts with zero value are omitted.\n"
     ":returns: {\"name\",0}"},
    {"_forceLazy", (PyCFunction)p4p_force_lazy, METH_NOARGS,
     "Force lazy initialization which might cause false positives"
     " leaks in differential ref counter testing."},
    {"serialize", (PyCFunction)p4p_serialize, METH_VARARGS|METH_KEYWORDS,
     "Serialize Type or Value to bytes"},
    {NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef p4pymodule = {
  PyModuleDef_HEAD_INIT,
    "_p4p",
    NULL,
    -1,
    P4P_methods,
};
#endif

extern "C" {
epicsShareFunc PyMOD(_p4p);
}

PyMOD(_p4p)
{
    try {
#if PY_MAJOR_VERSION >= 3
        PyRef mod(PyModule_Create(&p4pymodule));
#else
        PyRef mod(Py_InitModule("_p4p", P4P_methods));
#endif

        import_array();

        PyRef cancelled(PyErr_NewException("p4p.Cancelled", NULL, NULL));
        PyModule_AddObject(mod.get(), "Cancelled", cancelled.get());


        p4p_type_register(mod.get());
        p4p_value_register(mod.get());
        p4p_array_register(mod.get());
        p4p_server_register(mod.get());
        p4p_server_sharedpv_register(mod.get());
        p4p_server_provider_register(mod.get());
        p4p_client_register(mod.get());

        PyModule_AddIntConstant(mod.get(), "logLevelAll", epics::pvAccess::logLevelAll);
        PyModule_AddIntConstant(mod.get(), "logLevelTrace", epics::pvAccess::logLevelTrace);
        PyModule_AddIntConstant(mod.get(), "logLevelDebug", epics::pvAccess::logLevelDebug);
        PyModule_AddIntConstant(mod.get(), "logLevelInfo", epics::pvAccess::logLevelInfo);
        PyModule_AddIntConstant(mod.get(), "logLevelWarn", epics::pvAccess::logLevelWarn);
        PyModule_AddIntConstant(mod.get(), "logLevelError", epics::pvAccess::logLevelError);
        PyModule_AddIntConstant(mod.get(), "logLevelFatal", epics::pvAccess::logLevelFatal);
        PyModule_AddIntConstant(mod.get(), "logLevelOff", epics::pvAccess::logLevelOff);

        P4PCancelled = cancelled.release();

        MODINIT_RET(mod.release());
    } catch(std::exception& e) {
        PySys_WriteStderr("Import of _p4p failed: %s\n", e.what());
        MODINIT_RET(NULL);
    }
}

#ifdef TRACING
std::ostream& show_time(std::ostream& strm)
{
    timespec now;
    clock_gettime(CLOCK_REALTIME, &now);

    time_t sec = now.tv_sec;
    char buf[40];
    strftime(buf, sizeof(buf), "%H:%M:%S", localtime(&sec));
    size_t end = strlen(buf);
    PyOS_snprintf(buf+end, sizeof(buf)-end, ".%03u ", unsigned(now.tv_nsec/1000000u));
    buf[sizeof(buf)-1] = '\0';
    strm<<buf;
    return strm;
}
#endif // TRACING
