
#include <stddef.h>

#include "p4p.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace {

namespace pvd = epics::pvData;

struct Value {
    pvd::PVStructure::shared_pointer V;

    void storefld(epics::pvData::PVField *fld,
               const epics::pvData::Field *ftype,
               PyObject *obj);

    void store_struct(pvd::PVStructure* fld,
                      const pvd::Structure* ftype,
                      PyObject *obj);

    void store_union(pvd::PVUnion* fld,
                      const pvd::Union* ftype,
                      PyObject *obj);

    PyObject *fetchfld(pvd::PVField *fld,
                       const pvd::Field *ftype);
};

typedef PyClassWrapper<Value> P4PValue;

#define TRY P4PValue::reference_type SELF = P4PValue::unwrap(self); try

struct npmap {
    NPY_TYPES npy;
    pvd::ScalarType pvd;
};
const npmap np2pvd[] = {
    {NPY_BOOL, pvd::pvBoolean}, // bool stored as one byte
    {NPY_BYTE, pvd::pvByte},
    {NPY_SHORT, pvd::pvShort},
    {NPY_INT, pvd::pvInt},
    {NPY_LONG, pvd::pvLong},
    {NPY_UBYTE, pvd::pvUByte},
    {NPY_USHORT, pvd::pvUShort},
    {NPY_UINT, pvd::pvUInt},
    {NPY_ULONG, pvd::pvULong},
    {NPY_FLOAT, pvd::pvFloat},
    {NPY_DOUBLE, pvd::pvDouble},
    {NPY_NOTYPE}
};
NPY_TYPES ntype(pvd::ScalarType t) {
    for(const npmap *p = np2pvd; p->npy!=NPY_NOTYPE; p++) {
        if(p->pvd==t) return p->npy;
    }
    throw std::runtime_error(SB()<<"Unable to map scalar type '"<<(int)t<<"'");
}

//pvd::ScalarType ptype(NPY_TYPES t) {
//    for(const npmap *p = np2pvd; p->npy!=NPY_NOTYPE; p++) {
//        if(p->npy==t) return p->pvd;
//    }
//    throw std::runtime_error(SB()<<"Unable to map npy type '"<<(int)t<<"'");
//}


void Value::store_struct(pvd::PVStructure* fld,
                         const pvd::Structure* ftype,
                         PyObject *obj)
{
    const pvd::StringArray& names(ftype->getFieldNames());
    const pvd::FieldConstPtrArray& flds(ftype->getFields());
    const pvd::PVFieldPtrArray& vals(fld->getPVFields());

    size_t nfld = names.size();

    for(size_t i=0; i<nfld; i++) {
        PyRef name(PyUnicode_FromString(names[i].c_str()));

        PyRef item(PyObject_GetItem(obj, name.get()), allownull());
        if(!item.get()) {
            assert(PyErr_Occurred());
            if(!PyErr_ExceptionMatches(PyExc_KeyError))
                throw std::runtime_error("XXX");
            PyErr_Clear();
            continue;
        }

        storefld(vals[i].get(), flds[i].get(), item.get());
    }
}


void Value::store_union(pvd::PVUnion* fld,
                        const pvd::Union* ftype,
                        PyObject *obj)
{
    pvd::PVField::shared_pointer U;

    if(obj==Py_None) {
        // assign any Union w/ None to clear

        //BUG: can't clear variant union this way :P
        //F->select(pvd::PVUnion::UNDEFINED_INDEX);
        //BUG: can't clear variant this way either :{
        //fld->set(pvd::PVUnion::UNDEFINED_INDEX, pvd::PVFieldPtr());
        throw std::runtime_error("Bugs prevent clearing Variant Union");
        return;

    } else if(ftype->isVariant()) {
        // assign variant with plain value or wrapped Structure

        if(PyObject_TypeCheck(obj, &P4PValue::type)) {
            fld->set(pvd::PVUnion::UNDEFINED_INDEX, P4PValue_unwrap(obj));
            return;

        } else {
            // TODO: allow guess to be replaced
            pvd::Field::const_shared_pointer UT(P4PType_guess(obj));
            if(!UT)
                throw std::runtime_error(SB()<<"Unable to map "<<Py_TYPE(obj)->tp_name<<" for Variant Union storage");
            U = pvd::getPVDataCreate()->createPVField(UT);
            // fall down to assignment
        }

    } else if(PyTuple_Check(obj)) { // non-variant w/ explicit selection and value
        const char *select;
        PyObject *val;
        if(!PyArg_ParseTuple(obj, "sO;Assignment of non-variant union w/ (str, val).", &select, &val))
            throw std::runtime_error("XXX");

        if(val==Py_None) {
            //fld->set(pvd::PVUnion::UNDEFINED_INDEX, pvd::PVFieldPtr());
            throw std::runtime_error("Bugs prevent clearing non-Variant Union");
            return;

        } else if(PyObject_TypeCheck(val, &P4PValue::type)) {
            fld->set(select, P4PValue_unwrap(val));
            return;

        } else {
            U = fld->select(select);
            obj = val;
            // fall down to assignment
        }

    } else if(fld->getSelectedIndex()!=fld->UNDEFINED_INDEX) { // non-variant pre-selected
        U = fld->get();
        // fall down to assignment

    } else {
        // attempt "magic" selection.  (aka try each field until assignment succeeds...)
        for(size_t i=0, N=ftype->getNumberFields(); i<N; i++) {
            U = fld->select(i);
            try {
                storefld(U.get(),
                         U->getField().get(),
                         obj);
                return; // wow it worked
            } catch(std::runtime_error& e) {
                // try the next one
                if(i+1==N)
                    throw; // or not
                else if(PyErr_Occurred())
                    PyErr_Clear();
            }
        }
        throw std::runtime_error("Unable to automatically select non-Variant Union field");
    }

    storefld(U.get(),
             U->getField().get(),
             obj);

    fld->set(U);
}

void Value::storefld(pvd::PVField* fld,
                     const pvd::Field* ftype,
                     PyObject *obj)
{
    switch(ftype->getType()) {
    case pvd::scalar: {
        pvd::PVScalar* F = static_cast<pvd::PVScalar*>(fld);
        if(PyBool_Check(obj)) {
            F->putFrom<pvd::boolean>(obj==Py_True);
        } else if(PyInt_Check(obj)) {
            F->putFrom(PyInt_AsLong(obj));
        } else if(PyLong_Check(obj)) {
            F->putFrom(PyLong_AsLong(obj));
        } else if(PyFloat_Check(obj)) {
            F->putFrom(PyFloat_AsDouble(obj));
        } else if(PyBytes_Check(obj)) {
            std::string S(PyBytes_AS_STRING(obj), PyBytes_GET_SIZE(obj));
            F->putFrom(S);

        } else if(PyUnicode_Check(obj)) {
            PyRef S(PyUnicode_AsUTF8String(obj));

            std::string V(PyBytes_AS_STRING(S.get()), PyBytes_GET_SIZE(S.get()));
            F->putFrom(V);

        } else {
            throw std::runtime_error(SB()<<"Can't assign scalar field "<<fld->getFullName()<<" with "<<Py_TYPE(obj)->tp_name);
        }
    }
        return;
    case pvd::scalarArray: {
        pvd::PVScalarArray* F = static_cast<pvd::PVScalarArray*>(fld);
        const pvd::ScalarArray *T = static_cast<const pvd::ScalarArray *>(ftype);
        pvd::ScalarType etype = T->getElementType();

        if(etype==pvd::pvString) {
            PyRef iter(PyObject_GetIter(obj));

            pvd::shared_vector<std::string> vec;

            while(1) {
                PyRef I(PyIter_Next(iter.get()), allownull());
                if(!I.get()) {
                    if(PyErr_Occurred())
                        throw std::runtime_error("XXX");
                    break;
                }

                if(PyBytes_Check(I.get())) {
                    vec.push_back(std::string(PyBytes_AS_STRING(I.get()), PyBytes_GET_SIZE(I.get())));

                } else if(PyUnicode_Check(I.get())) {
                    PyRef B(PyUnicode_AsUTF8String(I.get()));

                    vec.push_back(std::string(PyBytes_AS_STRING(B.get()), PyBytes_GET_SIZE(B.get())));

                } else {
                    throw std::runtime_error(SB()<<"Can't assign string array element "<<fld->getFullName()<<" with "<<Py_TYPE(I.get())->tp_name);
                }
            }

            static_cast<pvd::PVStringArray*>(F)->replace(pvd::freeze(vec));

        } else {
            NPY_TYPES nptype(ntype(etype));

            PyRef V(PyArray_FromAny(obj, PyArray_DescrFromType(nptype), 0, 0,
                                    NPY_ARRAY_CARRAY_RO, NULL));

            if(PyArray_NDIM(V.get())!=1)
                throw std::runtime_error("Only 1-d array can be assigned");

            // TODO: detect reference cycles so we can avoid this copy
            //       Cycles can be created only if we both store and fetch
            //       by reference.
            pvd::shared_vector<void> buf(pvd::ScalarTypeFunc::allocArray(etype, PyArray_DIM(V.get(), 0)));

            memcpy(buf.data(), PyArray_DATA(V.get()), PyArray_NBYTES(V.get()));

            F->putFrom(pvd::freeze(buf));
        }
    }
        return;
    case pvd::structure: {
        pvd::PVStructure *F = static_cast<pvd::PVStructure*>(fld);
        const pvd::Structure *T = static_cast<const pvd::Structure*>(ftype);
        store_struct(F, T, obj);
    }
        return;
    case pvd::structureArray:
        // TODO
        break;
    case pvd::union_: {
        pvd::PVUnion* F = static_cast<pvd::PVUnion*>(fld);
        const pvd::Union *T = static_cast<const pvd::Union *>(ftype);

        store_union(F, T, obj);
    }
        return;
    case pvd::unionArray: {
        pvd::PVUnionArray* F = static_cast<pvd::PVUnionArray*>(fld);
        pvd::UnionConstPtr T = static_cast<const pvd::UnionArray *>(ftype)->getUnion();

        pvd::PVUnionArray::svector arr;
        PyRef iter(PyObject_GetIter(obj));

        pvd::PVDataCreatePtr create(pvd::getPVDataCreate());

        while(true) {
            PyRef item(PyIter_Next(iter.get()), allownull());
            if(!item.get()) {
                if(PyErr_Occurred())
                    throw std::runtime_error("XXX");
                break;
            }

            pvd::PVUnionPtr dest(create->createPVUnion(T));
            store_union(dest.get(), T.get(), item.get());

            arr.push_back(dest);
        }

        F->replace(pvd::freeze(arr));
    }
        return;
    }

    throw std::runtime_error("Storage of type not implemented");
}

PyObject *Value::fetchfld(pvd::PVField *fld,
                          const pvd::Field *ftype)
{
    switch(ftype->getType()) {
    case pvd::scalar: {
        pvd::PVScalar* F = static_cast<pvd::PVScalar*>(fld);
        const pvd::Scalar *T = static_cast<const pvd::Scalar*>(ftype);

        switch(T->getScalarType()) {
        case pvd::pvBoolean:
            return PyBool_FromLong(F->getAs<pvd::boolean>());
        case pvd::pvByte:
        case pvd::pvUByte:
        case pvd::pvShort:
        case pvd::pvUShort:
        case pvd::pvInt:
            return PyInt_FromLong(F->getAs<pvd::int32>());
        case pvd::pvUInt:
        case pvd::pvLong:
        case pvd::pvULong:
            return PyLong_FromLongLong(F->getAs<pvd::int64>());
        case pvd::pvFloat:
        case pvd::pvDouble:
            return PyFloat_FromDouble(F->getAs<double>());
        case pvd::pvString:
            return PyUnicode_FromString(F->getAs<std::string>().c_str());
        }
    }
        break;
    case pvd::scalarArray: {
        pvd::PVScalarArray* F = static_cast<pvd::PVScalarArray*>(fld);
        const pvd::ScalarArray *T = static_cast<const pvd::ScalarArray*>(ftype);
        pvd::ScalarType etype(T->getElementType());

        if(etype==pvd::pvString) {
            pvd::shared_vector<const std::string> arr(static_cast<pvd::PVStringArray*>(F)->view());

            PyRef list(PyList_New(arr.size()));

            for(size_t i=0; i<arr.size(); i++) {
                PyRef S(PyUnicode_FromString(arr[i].c_str()));

                PyList_SET_ITEM(list.get(), i, S.release());
            }

            return list.release();

        } else {
            NPY_TYPES npy(ntype(etype));

            pvd::shared_vector<const void> arr;
            F->getAs(arr);
            size_t esize = pvd::ScalarTypeFunc::elementSize(etype);
            npy_intp dim = arr.size()/esize;

            PyRef pyarr(PyArray_New(&PyArray_Type, 1, &dim, npy, NULL, (void*)arr.data(),
                                    esize, NPY_ARRAY_CARRAY_RO, NULL));

            PyObject *self = P4PValue::wrap(this);
            Py_INCREF(self);
            ((PyArrayObject*)pyarr.get())->base = self;

            return pyarr.release();
        }
    }
        break;
    case pvd::structure: {
        pvd::PVStructure* F = static_cast<pvd::PVStructure*>(fld);
        const pvd::Structure *T = static_cast<const pvd::Structure*>(ftype);

        const pvd::StringArray& names(T->getFieldNames());
        const pvd::FieldConstPtrArray& flds(T->getFields());
        const pvd::PVFieldPtrArray& vals(F->getPVFields());

        PyRef list(PyList_New(vals.size()));

        for(size_t i=0; i<vals.size(); i++) {
            PyRef val(fetchfld(vals[i].get(), flds[i].get()));

            PyRef item(Py_BuildValue("sO", names[i].c_str(), val.get()));

            PyList_SET_ITEM(list.get(), i, item.release());
        }

        return list.release();
    }
        break;
    case pvd::structureArray:
        break;
    case pvd::union_: {
        pvd::PVUnion* F = static_cast<pvd::PVUnion*>(fld);
        //const pvd::Union *T = static_cast<const pvd::Union *>(ftype);

        pvd::PVFieldPtr val(F->get());
        if(!val)
            Py_RETURN_NONE;
        else
            return fetchfld(val.get(), val->getField().get());
    }
        break;
    case pvd::unionArray: {
        pvd::PVUnionArray* F = static_cast<pvd::PVUnionArray*>(fld);

        pvd::PVUnionArray::const_svector arr(F->view());

        PyRef list(PyList_New(arr.size()));

        for(size_t i=0; i<arr.size(); i++) {
            PyRef ent;
            pvd::PVFieldPtr val;

            if(!arr[i] || !(val=arr[i]->get())) {
                ent.reset(Py_None, borrow());
            } else {
                ent.reset(fetchfld(val.get(), val->getField().get()));
            }

            PyList_SET_ITEM(list.get(), i, ent.release());
        }

        return list.release();
    }
        break;
    }
    throw std::runtime_error("map for read not implemented");
}

int P4PValue_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    TRY {
        const char *names[] = {"type", "value", "clone", NULL};
        PyObject *type = NULL, *value = NULL;
        PyObject *clone = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O!OO!", (char**)names,
                                        P4PType_type, &type,
                                        &value,
                                        P4PValue_type, &clone))
            return -1;

        if(SELF.V) {
            // magic construction w/ P4PValue_wrap()

        } else if(type) {
            pvd::Structure::const_shared_pointer S(P4PType_unwrap(type));

            pvd::PVStructure::shared_pointer V(pvd::getPVDataCreate()->createPVStructure(S));

            if(value) {
                SELF.store_struct(V.get(), S.get(), value);
            }

            SELF.V = V;

        } else if(clone) {
            SELF.V = P4PValue::unwrap(clone).V;

        } else {
            PyErr_SetString(PyExc_ValueError, "Value ctor requires type= or clone=");
            return -1;
        }

        return 0;
    }CATCH()
    return -1;
}

int P4PValue_setattr(PyObject *self, PyObject *name, PyObject *value)
{
    TRY {
        PyString S(name);
        pvd::PVFieldPtr fld = SELF.V->getSubField(S.str());
        if(!fld)
            return PyObject_GenericSetAttr((PyObject*)self, name, value);

        SELF.storefld(fld.get(),
                       fld->getField().get(),
                       value);

        return 0;
    }CATCH()
    return -1;

}

PyObject* P4PValue_getattr(PyObject *self, PyObject *name)
{
    TRY {
        PyString S(name);
        pvd::PVFieldPtr fld = SELF.V->getSubField(S.str());
        if(!fld)
            return PyObject_GenericGetAttr((PyObject*)self, name);

        if(fld->getField()->getType()==pvd::structure)
            return P4PValue_wrap(Py_TYPE(self), std::tr1::static_pointer_cast<pvd::PVStructure>(fld));

        return SELF.fetchfld(fld.get(),
                              fld->getField().get());
    }CATCH()
    return NULL;
}

PyObject* P4PValue_toList(PyObject *self, PyObject *args, PyObject *kwds)
{
    TRY {
        const char *names[] = {"name", NULL};
        const char *name = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|z", (char**)names, &name))
            return NULL;

        pvd::PVFieldPtr fld;
        if(name)
            fld = SELF.V->getSubField(name);
        else
            fld = SELF.V; // name==NULL converts entire structure

        if(!fld) {
            PyErr_SetString(PyExc_KeyError, name ? name : "<null>"); // should never actually be null
            return NULL;
        }

        return SELF.fetchfld(fld.get(),
                              fld->getField().get());

    }CATCH()
    return NULL;
}

PyObject *P4PValue_select(PyObject *self, PyObject *args, PyObject *kwds)
{
    TRY {
        const char *names[] = {"name", "selector", NULL};
        const char *name, *sel = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "sz", (char**)names, &name, &sel))
            return NULL;

        pvd::PVUnionPtr fld(SELF.V->getSubField<pvd::PVUnion>(name));
        if(!fld)
            return PyErr_Format(PyExc_KeyError, "%s", name);

        if(!sel) {
            fld->select(fld->UNDEFINED_INDEX);

        } else if(fld->getUnion()->isVariant()) {
            return PyErr_Format(PyExc_TypeError, "only select('fld') can be used to clear Variant Union");

        } else {
            fld->select(sel);
        }

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject *P4PValue_get(PyObject *self, PyObject *args, PyObject *kwds)
{
    TRY {
        const char *name;
        PyObject *defval = Py_None;
        if(!PyArg_ParseTuple(args, "s|O", &name, &defval))
            return NULL;

        pvd::PVFieldPtr fld = SELF.V->getSubField(name);
        if(!fld) {
            Py_INCREF(defval);
            return defval;
        }

        if(fld->getField()->getType()==pvd::structure)
            return P4PValue_wrap(Py_TYPE(self), std::tr1::static_pointer_cast<pvd::PVStructure>(fld));

        return SELF.fetchfld(fld.get(),
                              fld->getField().get());
    }CATCH()
    return NULL;
}

Py_ssize_t P4PValue_len(PyObject *self)
{
    TRY {
        return SELF.V->getNumberFields();
    }CATCH()
    return -1;
}

int P4PValue_setitem(PyObject *self, PyObject *name, PyObject *value)
{
    TRY {
        PyString S(name);
        pvd::PVFieldPtr fld = SELF.V->getSubField(S.str());
        if(!fld) {
            PyErr_SetString(PyExc_KeyError, S.str().c_str());
            return -1;
        }

        SELF.storefld(fld.get(),
                       fld->getField().get(),
                       value);

        return 0;
    }CATCH()
    return -1;

}

PyObject* P4PValue_getitem(PyObject *self, PyObject *name)
{
    TRY {
        PyString S(name);
        pvd::PVFieldPtr fld = SELF.V->getSubField(S.str());
        if(!fld) {
            PyErr_SetString(PyExc_KeyError, S.str().c_str());
            return NULL;
        }

        if(fld->getField()->getType()==pvd::structure)
            return P4PValue_wrap(Py_TYPE(self), std::tr1::static_pointer_cast<pvd::PVStructure>(fld));

        return SELF.fetchfld(fld.get(),
                              fld->getField().get());
    }CATCH()
    return NULL;
}

PyMappingMethods P4PValue_mapping = {
    (lenfunc)&P4PValue_len,
    (binaryfunc)&P4PValue_getitem,
    (objobjargproc)&P4PValue_setitem
};

static PyMethodDef P4PValue_methods[] = {
    {"tolist", (PyCFunction)&P4PValue_toList, METH_VARARGS|METH_KEYWORDS,
     "Transform wrapped Structure into a list of tuples."},
    {"select", (PyCFunction)&P4PValue_select, METH_VARARGS|METH_KEYWORDS,
     "pre-select/clear Union"},
    {"get", (PyCFunction)&P4PValue_get, METH_VARARGS,
     "Fetch a field value, or a default if it does not exist"},
    {NULL}
};

template<>
PyTypeObject P4PValue::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_p4p.Value",
    sizeof(P4PValue),
};

} // namespace

PyTypeObject* P4PValue_type = &P4PValue::type;

void p4p_value_register(PyObject *mod)
{
    P4PValue::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    P4PValue::type.tp_new = &P4PValue::tp_new;
    P4PValue::type.tp_init = &P4PValue_init;
    P4PValue::type.tp_dealloc = &P4PValue::tp_dealloc;
    P4PValue::type.tp_getattro = &P4PValue_getattr;
    P4PValue::type.tp_setattro = &P4PValue_setattr;

    P4PValue::type.tp_as_mapping = &P4PValue_mapping;

    P4PValue::type.tp_methods = P4PValue_methods;

    //P4PValue::type.tp_weaklistoffset = offsetof()

    if(PyType_Ready(&P4PValue::type))
        throw std::runtime_error("failed to initialize P4PValue_type");

    Py_INCREF((PyObject*)&P4PValue::type);
    if(PyModule_AddObject(mod, "Value", (PyObject*)&P4PValue::type)) {
        Py_DECREF((PyObject*)&P4PValue::type);
        throw std::runtime_error("failed to add _p4p.Value");
    }
}

epics::pvData::PVStructure::shared_pointer P4PValue_unwrap(PyObject *obj)
{
    if(!PyObject_TypeCheck(obj, &P4PValue::type))
        throw std::runtime_error("Not a _p4p.Value");
    return P4PValue::unwrap(obj).V;
}

PyObject *P4PValue_wrap(PyTypeObject *type, const epics::pvData::PVStructure::shared_pointer& V)
{
    assert(V.get());
    if(!PyType_IsSubtype(type, &P4PValue::type))
        throw std::runtime_error("Not a sub-class of _p4p.Value");

    // magic construction of potentially derived type...

    PyRef args(PyTuple_New(0));
    PyRef kws(PyDict_New());

    PyRef ret(type->tp_new(type, args.get(), kws.get()));

    // inject value *before* __init__ of base or derived type runs
    P4PValue::unwrap(ret.get()).V = V;

    if(type->tp_init(ret.get(), args.get(), kws.get()))
        throw std::runtime_error("XXX");

    return ret.release();
}
