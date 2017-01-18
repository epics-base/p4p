
#include <stddef.h>

#include "p4p.h"

namespace {

namespace pvd = epics::pvData;

typedef PyClassWrapper<pvd::Structure::const_shared_pointer> P4PType;

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
        if(!PyArg_ParseTuple(ent.get(), "sO;Expected list of tuples (str, object) or (str, str, object)", &key, &val))
            throw std::runtime_error("XXX");

        if(0) {}
#if PY_MAJOR_VERSION < 3
        else if(PyBytes_Check(val)) {
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

int P4PType_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *spec;
    const char *id = NULL;
    static const char *names[] = {"spec", "id", NULL};
    TRY {
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|z", (char**)names, &spec, &id))
            return -1;

        pvd::FieldBuilderPtr builder(pvd::getFieldCreate()->createFieldBuilder());
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

PyObject* P4PType_aspy(PyObject *self) {
    TRY {
        assert(SELF.get());

        const pvd::StringArray& names(SELF->getFieldNames());
        const pvd::FieldConstPtrArray& flds(SELF->getFields());

        PyRef list(struct2py(names, flds));
        std::string id(SELF->getID());

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

int P4PType_traverse(PyObject *self, visitproc visit, void *arg)
{
    return 0;
}

int P4PType_clear(PyObject *self)
{
    return 0;
}

template<>
PyTypeObject P4PType::type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_p4p.Type",
    sizeof(P4PType),
};

} // namespace

PyTypeObject* P4PType_type = &P4PType::type;

void p4p_type_register(PyObject *mod)
{
    P4PType::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC;
    P4PType::type.tp_new = &P4PType::tp_new;
    P4PType::type.tp_init = &P4PType_init;
    P4PType::type.tp_dealloc = &P4PType::tp_dealloc;
    P4PType::type.tp_traverse = &P4PType_traverse;
    P4PType::type.tp_clear = &P4PType_clear;

    P4PType::type.tp_methods = P4PType_members;

    P4PType::type.tp_weaklistoffset = offsetof(P4PType, weak);

    if(PyType_Ready(&P4PType::type))
        throw std::runtime_error("failed to initialize P4PType_type");

    Py_INCREF((PyObject*)&P4PType::type);
    if(PyModule_AddObject(mod, "Type", (PyObject*)&P4PType::type)) {
        Py_DECREF((PyObject*)&P4PType::type);
        throw std::runtime_error("failed to add _p4p.Type");
    }
}

pvd::Structure::const_shared_pointer P4PType_unwrap(PyObject *obj)
{
    return P4PType::unwrap(obj);
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
    // TODO: guess for ndarray, list, and dict
}
