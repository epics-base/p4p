
#include <stddef.h>

#include "p4p.h"

namespace {

namespace pvd = epics::pvData;

struct P4PType {
    PyObject_HEAD

    PyObject *weak;

    // we are playing some (I think safe) games here.
    // all non-POD types must appear in sub-struct C
    struct C_t {
        pvd::Structure::const_shared_pointer S;
    } C;
    pvd::Structure::const_shared_pointer& S() { return C.S; }
};

PyObject* P4PType_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    try {
        // we use python alloc instead of new here so that we could participate in GC
        PyRef self(type->tp_alloc(type, 0));
        P4PType *SELF = (P4PType*)self.get();

        SELF->weak = NULL;

        // The following can zero out the PyObject_HEAD members
        //new (self.get()) P4PType();
        // instead we only C++ initialize the sub-struct C
        new (&SELF->C) P4PType::C_t();

        return self.release();
    } CATCH()
    return NULL;
}

void P4PType_dtor(P4PType *self)
{
    try {
        self->C.~C_t();
    } CATCH()
    Py_TYPE(self)->tp_free((PyObject*)self);
}

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
 *   ('field', ('s', 'name'|None, [...])  # sub-struct
 *   ('field', ('u', None, [...])),       # union
 *   ('field', ('as', 'name'|None, [...]) # sub-struct array
 *   ('field', ('au', None, [...])),      # union array
 * ]
 */
void py2struct(pvd::FieldBuilderPtr& builder, PyObject *o)
{
    PyRef iter(PyObject_GetIter(o));

    while(true) {
        PyRef ent(PyIter_Next(iter.get()), allownull());
        if(!ent.get()) {
            if(PyErr_Occurred())
                throw std::runtime_error("XXX"); // exception already set
            break;
        }

        const char *key;
        PyObject *val;
        if(!PyArg_ParseTuple(ent.get(), "sO;Expected list of tuples (str, object)", &key, &val))
            throw std::runtime_error("XXX");

        if(0) {}
#if PY_MAJOR_VERSION < 3
        else if(PyString_Check(val)) {
            const char *spec = PyString_AsString(val);
            py2struct_plain(builder, key, spec);
#endif
        } else if(PyUnicode_Check(val)) {
            PyRef str(PyUnicode_AsASCIIString(val));
            const char *spec = PyBytes_AsString(val);
            py2struct_plain(builder, key, spec);

        } else {
            const char *tkey, *tname;
            PyObject *members;
            if(!PyArg_ParseTuple(val, "szO;Expected list of tuples (str, str, object)", &tkey, &tname, &members))
                throw std::runtime_error("XXX");

            if(tkey[0]=='a') {
                if(tkey[1]=='s') {
                    builder = builder->addNestedStructureArray(key);
                } else if(tkey[1]=='u') {
                    builder = builder->addNestedUnionArray(key);
                } else {
                    throw std::runtime_error(SB()<<"Unknown spec \""<<tkey<<"\"");
                }
            } else if(tkey[0]=='s') {
                builder = builder->addNestedStructure(key);
            } else if(tkey[0]=='u') {
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

int P4PType_init(P4PType *self, PyObject *args, PyObject *kwds)
{
    PyObject *spec;
    const char *id = NULL;
    static const char *names[] = {"spec", "id", NULL};
    try {
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|z", (char**)names, &spec, &id))
            return -1;

        pvd::FieldBuilderPtr builder(pvd::getFieldCreate()->createFieldBuilder());
        if(id)
            builder->setId(id);
        py2struct(builder, spec);
        self->S() = builder->createStructure();

        if(!self->S().get()) {
            PyErr_SetString(PyExc_ValueError, "Spec did not result in Structure");
            return -1;
        }

        return 0;
    }CATCH()
    return -1;
}

PyObject* struct2py(const pvd::StringArray& names,
                    const pvd::FieldConstPtrArray& flds)
{
    assert(names.size()==flds.size());

    const size_t nfld = names.size();
    PyRef list(PyList_New(nfld));

    for(size_t i=0; i<nfld; i++) {
        PyRef value;

        pvd::Type ftype = flds[i]->getType();

        switch(ftype) {
        case pvd::scalar: {
            pvd::ScalarType stype = static_cast<const pvd::Scalar*>(flds[i].get())->getScalarType();
            char spec[2] = {sname(stype), '\0'};

            value.reset(Py_BuildValue("ss", names[i].c_str(), spec));
        }
            break;
        case pvd::scalarArray: {
            pvd::ScalarType stype = static_cast<const pvd::ScalarArray*>(flds[i].get())->getElementType();
            char spec[3] = {'a', sname(stype), '\0'};
            PyRef str(PyUnicode_FromString(spec));
            value.reset(Py_BuildValue("ss", names[i].c_str(), spec));
        }
            break;
        case pvd::structure: {
            pvd::StructureConstPtr S(std::tr1::static_pointer_cast<const pvd::Structure>(flds[i]));

            const pvd::StringArray& subnames(S->getFieldNames());
            const pvd::FieldConstPtrArray& subflds(S->getFields());
            PyRef members(struct2py(subnames, subflds));

            std::string id(S->getID());

            value.reset(Py_BuildValue("s(szO)", names[i].c_str(), "s",
                                      id.empty() ? NULL : id.c_str(),
                                      members.get()
                                      ));
        }
            break;
        case pvd::structureArray: {
            pvd::StructureConstPtr S(std::tr1::static_pointer_cast<const pvd::StructureArray>(flds[i])->getStructure());

            const pvd::StringArray& subnames(S->getFieldNames());
            const pvd::FieldConstPtrArray& subflds(S->getFields());
            PyRef members(struct2py(subnames, subflds));

            std::string id(S->getID()); // TODO: which ID?
            value.reset(Py_BuildValue("s(szO)", names[i].c_str(), "as",
                                      id.empty() ? NULL : id.c_str(),
                                      members.get()
                                      ));
        }
            break;
        case pvd::union_: {
            pvd::UnionConstPtr S(std::tr1::static_pointer_cast<const pvd::Union>(flds[i]));

            if(S->isVariant()) {
                value.reset(Py_BuildValue("ss", names[i].c_str(), "v"));

            } else {
                const pvd::StringArray& subnames(S->getFieldNames());
                const pvd::FieldConstPtrArray& subflds(S->getFields());
                PyRef members(struct2py(subnames, subflds));

                std::string id(S->getID());
                value.reset(Py_BuildValue("s(szO)", names[i].c_str(), "u",
                                          id.empty() ? NULL : id.c_str(),
                                          members.get()
                                          ));
            }
        }
            break;
        case pvd::unionArray: {
            pvd::UnionConstPtr S(std::tr1::static_pointer_cast<const pvd::UnionArray>(flds[i])->getUnion());

            if(S->isVariant()) {
                value.reset(Py_BuildValue("ss", names[i].c_str(), "av"));

            } else {
                const pvd::StringArray& subnames(S->getFieldNames());
                const pvd::FieldConstPtrArray& subflds(S->getFields());
                PyRef members(struct2py(subnames, subflds));

                std::string id(S->getID()); // TODO: which ID?
                value.reset(Py_BuildValue("s(szO)", names[i].c_str(), "au",
                                          id.empty() ? NULL : id.c_str(),
                                          members.get()
                                          ));
            }
        }
            break;
        }

        if(!value.get())
            throw std::runtime_error(SB()<<"Unable to translate \""<<names[i]<<"\"");
        PyList_SET_ITEM(list.get(), i, value.release());
    }

    return list.release();
}

PyObject* P4PType_aspy(P4PType *self) {
    try {
        assert(self->S().get());

        const pvd::StringArray& names(self->S()->getFieldNames());
        const pvd::FieldConstPtrArray& flds(self->S()->getFields());

        PyRef list(struct2py(names, flds));
        std::string id(self->S()->getID());

        return Py_BuildValue("szO", "s",
                             id.empty() ? NULL : id.c_str(),
                             list.get());
    } CATCH()
    return NULL;
}

static struct PyMethodDef P4PType_members[] = {
    {"aspy", (PyCFunction)P4PType_aspy, METH_NOARGS,
     "Return spec for this PVD Structure"},
    {NULL}
};

int P4PType_traverse(P4PType *self, visitproc visit, void *arg)
{
    return 0;
}

int P4PType_clear(P4PType *self)
{
    return 0;
}

} // namespace

PyTypeObject P4PType_type = {
    #if PY_MAJOR_VERSION >= 3
        PyVarObject_HEAD_INIT(NULL, 0)
    #else
        PyObject_HEAD_INIT(NULL)
        0,
    #endif
    "_p4p.Type",
    sizeof(P4PType),
};

void p4p_type_register(PyObject *mod)
{
    P4PType_type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    P4PType_type.tp_new = (newfunc)&P4PType_new;
    P4PType_type.tp_init = (initproc)&P4PType_init;
    P4PType_type.tp_dealloc = (destructor)&P4PType_dtor;
    P4PType_type.tp_traverse = (traverseproc)&P4PType_traverse;
    P4PType_type.tp_clear = (inquiry)&P4PType_clear;

    P4PType_type.tp_methods = P4PType_members;

    P4PType_type.tp_weaklistoffset = offsetof(P4PType, weak);

    if(PyType_Ready(&P4PType_type))
        throw std::runtime_error("failed to initialize P4PType_type");

    Py_INCREF((PyObject*)&P4PType_type);
    if(PyModule_AddObject(mod, "Type", (PyObject*)&P4PType_type)) {
        Py_DECREF((PyObject*)&P4PType_type);
        throw std::runtime_error("failed to add _p4p.Type");
    }
}

pvd::Structure::const_shared_pointer P4PType_unwrap(PyObject *obj)
{
    if(Py_TYPE(obj)!=&P4PType_type)
        throw std::runtime_error("Not a _p4p.Type");
    P4PType *ptype = (P4PType*)obj;
    return ptype->S();
}

PyObject *P4PType_wrap(const pvd::Structure::const_shared_pointer& s)
{
    assert(s.get());
    PyRef ret(P4PType_new(&P4PType_type, NULL, NULL));
    ((P4PType*)ret.get())->S() = s;
    return ret.release();
}

epics::pvData::Field::const_shared_pointer P4PType_guess(PyObject *obj)
{
    pvd::FieldCreatePtr create(pvd::getFieldCreate());

    if(0) {
    } else if(PyInt_Check(obj)) {
        return create->createScalar(pvd::pvInt);
    } else if(PyLong_Check(obj)) {
        return create->createScalar(pvd::pvLong);
    } else if(PyFloat_Check(obj)) {
        return create->createScalar(pvd::pvDouble);
    } else if(PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        return create->createScalar(pvd::pvString);
    } else {
        return epics::pvData::Field::const_shared_pointer();
    }
}
