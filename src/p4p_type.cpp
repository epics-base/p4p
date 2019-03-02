
#include <stddef.h>

#include "p4p.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace pvd = epics::pvData;

typedef PyClassWrapper<pvd::Structure::const_shared_pointer> P4PType;

PyClassWrapper_DEF(P4PType, "Type")

namespace {

#define TRY P4PType::reference_type SELF = P4PType::unwrap(self); try

struct c2t {
    char c;
    pvd::ScalarType t;
};
const c2t plainmap[] = {
    {'?', pvd::pvBoolean},
    {'s', pvd::pvString},
    {'b', pvd::pvByte},
    {'h', pvd::pvShort},
    {'i', pvd::pvInt},
    {'l', pvd::pvLong},
    {'B', pvd::pvUByte},
    {'H', pvd::pvUShort},
    {'I', pvd::pvUInt},
    {'L', pvd::pvULong},
    {'f', pvd::pvFloat},
    {'d', pvd::pvDouble},
    {'\0'}
};

pvd::ScalarType stype(char c) {
    for(const c2t *p = plainmap; p->c; p++) {
        if(p->c==c) return p->t;
    }
    throw std::runtime_error(SB()<<"Unable to map spec '"<<(int)c<<"'");
}

char sname(pvd::ScalarType t) {
    for(const c2t *p = plainmap; p->c; p++) {
        if(p->t==t) return p->c;
    }
    throw std::runtime_error(SB()<<"Unable to map type '"<<(int)t<<"'");
}

void py2struct_plain(pvd::FieldBuilderPtr& builder, const char *key, const char *spec)
{
    if(spec[0]=='a') {
        if(spec[1]=='v') {
            builder->add(key, pvd::getFieldCreate()->createVariantUnionArray());

        } else {
            builder->addArray(key, stype(spec[1]));
        }

    } else if(spec[0]=='v') {
        builder->add(key, pvd::getFieldCreate()->createVariantUnion());

    } else {
        builder->add(key, stype(spec[0]));
    }
}

/** Translate python type spect
 *
 * [
 *   ('field', 'typename'),               # simple types
 *   ('field', ('S', 'name'|None, [...])  # sub-struct
 *   ('field', ('U', None, [...])),       # union
 *   ('field', ('aS', 'name'|None, [...]) # sub-struct array
 *   ('field', ('aU', None, [...])),      # union array
 * ]
 */
void py2struct(pvd::FieldBuilderPtr& builder, PyObject *o)
{
    PyRef iter(PyObject_GetIter(o));

    while(true) {
        PyRef ent(PyIter_Next(iter.get()), nextiter());
        if(!ent) break;

        const char *key;
        PyObject *val;
        if(!PyArg_ParseTuple(ent.get(), "sO;Expected list of tuples eg. (str, object) or (str, ('S', str, object))", &key, &val))
            throw std::runtime_error("XXX");

        if(0) {
        } else if(PyObject_IsInstance(val, (PyObject*)&P4PType::type)) {
            pvd::StructureConstPtr sub(P4PType_unwrap(val));

            builder->add(key, sub);

#if PY_MAJOR_VERSION < 3
        } else if(PyBytes_Check(val)) {
            const char *spec = PyBytes_AsString(val);
            py2struct_plain(builder, key, spec);
#endif
        } else if(PyUnicode_Check(val)) {
            PyRef str(PyUnicode_AsASCIIString(val));
            const char *spec = PyBytes_AsString(str.get());
            py2struct_plain(builder, key, spec);

        } else {
            const char *tkey, *tname;
            PyObject *members;
            if(!PyArg_ParseTuple(val, "szO;Expected list of tuples (str, str, object) or (str, object)", &tkey, &tname, &members))
                throw std::runtime_error("XXX");

            if(tkey[0]=='a') {
                if(tkey[1]=='s' || tkey[1]=='S') {
                    builder = builder->addNestedStructureArray(key);
                } else if(tkey[1]=='u' || tkey[1]=='U') {
                    builder = builder->addNestedUnionArray(key);
                } else {
                    throw std::runtime_error(SB()<<"Unknown spec \""<<tkey<<"\"");
                }
            } else if(tkey[0]=='s' || tkey[0]=='S') {
                builder = builder->addNestedStructure(key);
            } else if(tkey[0]=='u' || tkey[0]=='U') {
                builder = builder->addNestedUnion(key);
            } else {
                throw std::runtime_error(SB()<<"Unknown spec \""<<tkey<<"\"");
            }

            if(tname)
                builder->setId(tname);

            py2struct(builder, members);

            builder = builder->endNested();
        }
    }
}

int P4PType_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *spec;
    const char *id = NULL;
    PyObject *base = Py_None;
    static const char *names[] = {"spec", "id", "base", NULL};
    TRY {
        if(SELF.get())
            return 0; // magic case when called from P4PType_wrap()

        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|zO!", (char**)names, &spec, &id, (PyObject*)&P4PType::type, &base))
            return -1;

        pvd::FieldBuilderPtr builder;
        if(base==Py_None)
            builder = pvd::getFieldCreate()->createFieldBuilder();
        else
            builder = pvd::getFieldCreate()->createFieldBuilder(P4PType::unwrap(base));
        if(id)
            builder->setId(id);
        py2struct(builder, spec);
        SELF = builder->createStructure();

        if(!SELF.get()) {
            PyErr_SetString(PyExc_ValueError, "Spec did not result in Structure");
            return -1;
        }

        return 0;
    }CATCH()
    return -1;
}

PyObject* struct2py(const pvd::StringArray& names,
                    const pvd::FieldConstPtrArray& flds);

PyObject* field2py(const pvd::FieldConstPtr& fld)
{
    switch(fld->getType()) {
    case pvd::scalar: {
        pvd::ScalarType stype = static_cast<const pvd::Scalar*>(fld.get())->getScalarType();
        char spec[2] = {sname(stype), '\0'};

        return Py_BuildValue("s", spec);
    }
        break;
    case pvd::scalarArray: {
        pvd::ScalarType stype = static_cast<const pvd::ScalarArray*>(fld.get())->getElementType();
        char spec[3] = {'a', sname(stype), '\0'};
        return Py_BuildValue("s", spec);
    }
        break;
    case pvd::structure: {
        pvd::StructureConstPtr S(std::tr1::static_pointer_cast<const pvd::Structure>(fld));

        const pvd::StringArray& subnames(S->getFieldNames());
        const pvd::FieldConstPtrArray& subflds(S->getFields());
        PyRef members(struct2py(subnames, subflds));

        std::string id(S->getID());

        return Py_BuildValue("szO", "S",
                             id.empty() ? NULL : id.c_str(),
                             members.get());
    }
        break;
    case pvd::structureArray: {
        pvd::StructureConstPtr S(std::tr1::static_pointer_cast<const pvd::StructureArray>(fld)->getStructure());

        const pvd::StringArray& subnames(S->getFieldNames());
        const pvd::FieldConstPtrArray& subflds(S->getFields());
        PyRef members(struct2py(subnames, subflds));

        std::string id(S->getID()); // TODO: which ID?
        return Py_BuildValue("szO", "aS",
                             id.empty() ? NULL : id.c_str(),
                             members.get());
    }
        break;
    case pvd::union_: {
        pvd::UnionConstPtr S(std::tr1::static_pointer_cast<const pvd::Union>(fld));

        if(S->isVariant()) {
            return Py_BuildValue("s", "v");

        } else {
            const pvd::StringArray& subnames(S->getFieldNames());
            const pvd::FieldConstPtrArray& subflds(S->getFields());
            PyRef members(struct2py(subnames, subflds));

            std::string id(S->getID());
            return Py_BuildValue("szO", "U",
                                 id.empty() ? NULL : id.c_str(),
                                 members.get());
        }
    }
        break;
    case pvd::unionArray: {
        pvd::UnionConstPtr S(std::tr1::static_pointer_cast<const pvd::UnionArray>(fld)->getUnion());

        if(S->isVariant()) {
            return Py_BuildValue("s", "av");

        } else {
            const pvd::StringArray& subnames(S->getFieldNames());
            const pvd::FieldConstPtrArray& subflds(S->getFields());
            PyRef members(struct2py(subnames, subflds));

            std::string id(S->getID()); // TODO: which ID?
            return Py_BuildValue("szO", "aU",
                                 id.empty() ? NULL : id.c_str(),
                                 members.get());
        }
    }
        break;
    }

    return PyErr_Format(PyExc_RuntimeError, "field2py() unsupported field type");
}

PyObject* struct2py(const pvd::StringArray& names,
                    const pvd::FieldConstPtrArray& flds)
{
    assert(names.size()==flds.size());

    const size_t nfld = names.size();
    PyRef list(PyList_New(nfld));

    for(size_t i=0; i<nfld; i++) {

        PyRef value(field2py(flds[i]));
        PyRef pair(Py_BuildValue("sO", names[i].c_str(), value.get()));
        value.release();

        PyList_SET_ITEM(list.get(), i, pair.release());
    }

    return list.release();
}

PyObject* P4PType_id(PyObject *self) {
    TRY {
        assert(SELF.get());

        return PyUnicode_FromString(SELF->getID().c_str());
    }CATCH()
    return NULL;
}

PyObject* P4PType_aspy(PyObject *self, PyObject *args, PyObject *kws) {
    TRY {
        static const char *names[] = {"name", NULL};
        const char *fname = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|z", (char**)names, &fname))
            return NULL;

        assert(SELF.get());

        pvd::FieldConstPtr fld(fname ? SELF->getField(fname) : pvd::FieldConstPtr(SELF));

        return field2py(fld);
    } CATCH()
    return NULL;
}

PyObject* P4PType_has(PyObject *self, PyObject *args, PyObject *kws) {
    TRY {
        static const char *names[] = {"name", "type", NULL};
        const char *fname;
        PyObject *ftype = Py_None;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "s|O", (char**)names, &fname, &ftype))
            return NULL;

        pvd::Field::const_shared_pointer fld(SELF->getField(fname));
        if(!fld)
            Py_RETURN_FALSE;

        if(ftype!=Py_None) {
            return PyErr_Format(PyExc_NotImplementedError, "field type matching not implemented");
        }

        Py_RETURN_TRUE;
    } CATCH()
    return NULL;
}

PyObject* P4PType_keys(PyObject *self)
{
    TRY {
        const pvd::StringArray& names(SELF->getFieldNames());

        PyRef list(PyList_New(names.size()));

        for(size_t i=0; i<names.size(); i++) {
            PyObject *name = PyUnicode_FromString(names[i].c_str());
            if(!name)
                return NULL;
            PyList_SET_ITEM(list.get(), i, name);
        }

        return list.release();
    } CATCH()
    return NULL;
}

PyObject* P4PType_magic(PyObject *self, PyObject *args)
{
    try {
        PyObject *replacement, *old;
        if(!PyArg_ParseTuple(args, "O", &replacement))
            return NULL;

        if(!PyObject_IsSubclass(replacement, (PyObject*)&P4PType::type))
            return PyErr_Format(PyExc_TypeError, "Not sub-class");

        old = (PyObject*)P4PType_type;
        P4PType_type = (PyTypeObject*)replacement;

        Py_INCREF(replacement);
        Py_DECREF(old);

        Py_RETURN_NONE;
    } CATCH()
    return NULL;
}

Py_ssize_t P4PType_len(PyObject *self)
{
    TRY {
        return SELF->getNumberFields();
    }CATCH()
    return -1;
}

PyObject* P4PType_getitem(PyObject *self, PyObject *name)
{
    TRY {
        PyString S(name);
        pvd::FieldConstPtr fld(SELF->getField(S.str()));
        if(!fld) {
            return PyErr_Format(PyExc_KeyError, "%s", S.str().c_str());
        } else if(fld->getType()!=pvd::structure) {
            return field2py(fld);
        } else {
            return P4PType_wrap(P4PType_type, std::tr1::static_pointer_cast<const pvd::Structure>(fld));
        }
    }CATCH()
    return NULL;
}

PyMappingMethods P4PType_mapping = {
    (lenfunc)&P4PType_len,
    (binaryfunc)&P4PType_getitem,
    NULL,
};

static struct PyMethodDef P4PType_members[] = {
    {"getID", (PyCFunction)P4PType_id, METH_NOARGS,
     "Return Structure ID"},
    {"keys", (PyCFunction)P4PType_keys, METH_NOARGS,
     "Return field names"},
    {"aspy", (PyCFunction)P4PType_aspy, METH_VARARGS|METH_KEYWORDS,
     "Return spec for this PVD Structure"},
    {"has", (PyCFunction)P4PType_has, METH_VARARGS|METH_KEYWORDS,
     "has('name', type=None)\n\nTest structure member presence"},
    {"_magic", (PyCFunction)P4PType_magic, METH_VARARGS|METH_STATIC,
     "Don't call this!"},
    {NULL}
};

int P4PType_traverse(PyObject *self, visitproc visit, void *arg)
{
    return 0;
}

int P4PType_clear(PyObject *self)
{
    return 0;
}

} // namespace

PyTypeObject* P4PType_type = &P4PType::type;

void p4p_type_register(PyObject *mod)
{
    P4PType::buildType();
    P4PType::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC;
    P4PType::type.tp_init = &P4PType_init;
    P4PType::type.tp_traverse = &P4PType_traverse;
    P4PType::type.tp_clear = &P4PType_clear;

    P4PType::type.tp_as_mapping = &P4PType_mapping;

    P4PType::type.tp_methods = P4PType_members;

    P4PType::finishType(mod, "TypeBase");
}

PyObject* P4PType_wrap(PyTypeObject *type, const epics::pvData::Structure::const_shared_pointer& S)
{
    assert(S.get());
    if(!PyType_IsSubtype(type, &P4PType::type))
        throw std::runtime_error("Not a sub-class of _p4p.TypeBase");

    // magic construction of potentially derived type...

    PyRef args(PyTuple_New(0));
    PyRef kws(PyDict_New());

    PyRef ret(type->tp_new(type, args.get(), kws.get()));

    // inject value *before* __init__ of base or derived type runs
    P4PType::unwrap(ret.get()) = S;

    if(type->tp_init(ret.get(), args.get(), kws.get()))
        throw std::runtime_error("XXX");

    return ret.release();

}

pvd::Structure::const_shared_pointer P4PType_unwrap(PyObject *obj)
{
    return P4PType::unwrap(obj);
}

epics::pvData::Field::const_shared_pointer P4PType_guess(PyObject *obj)
{
    pvd::FieldCreatePtr create(pvd::getFieldCreate());

    if(0) {
    } else if(PyBool_Check(obj)) {
        return create->createScalar(pvd::pvBoolean);
#if PY_MAJOR_VERSION < 3
    } else if(PyInt_Check(obj)) {
        return create->createScalar(pvd::pvInt);
#endif
    } else if(PyLong_Check(obj)) {
        return create->createScalar(pvd::pvLong);
    } else if(PyFloat_Check(obj)) {
        return create->createScalar(pvd::pvDouble);
    } else if(PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        return create->createScalar(pvd::pvString);
    } else if(PyList_Check(obj)) {
        return create->createScalarArray(pvd::pvString);
    } else if(PyArray_Check(obj)) {
        switch(PyArray_TYPE(obj)) {
#define CASE(NTYPE, PTYPE) case NTYPE: return create->createScalarArray(PTYPE);
        CASE(NPY_BOOL, pvd::pvBoolean) // bool stored as one byte
        CASE(NPY_INT8, pvd::pvByte)
        CASE(NPY_INT16, pvd::pvShort)
        CASE(NPY_INT32, pvd::pvInt)
        CASE(NPY_INT64, pvd::pvLong)
        CASE(NPY_UINT8, pvd::pvUByte)
        CASE(NPY_UINT16, pvd::pvUShort)
        CASE(NPY_UINT32, pvd::pvUInt)
        CASE(NPY_UINT64, pvd::pvULong)
        CASE(NPY_FLOAT, pvd::pvFloat)
        CASE(NPY_DOUBLE, pvd::pvDouble)
        CASE(NPY_STRING, pvd::pvString)
#undef CASE
        }
    }
    return epics::pvData::Field::const_shared_pointer();
    // TODO: guess for  list and dict
}
