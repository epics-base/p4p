
#include <stddef.h>

#include "p4p.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL P4P_PyArray_API
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#ifndef HAVE_LONG_LONG
#  error Require long long
#endif

namespace {

namespace pvd = epics::pvData;

struct Value {
    // structure we are wrapping
    pvd::PVStructure::shared_pointer V;
    // which fields of this structure have been initialized w/ non-default values
    // NULL when not tracking, treated as bit 0 set (aka all initialized)
    pvd::BitSet::shared_pointer I;

    // assignment of PVStructure from Object

    void storefld(epics::pvData::PVField *fld,
               const epics::pvData::Field *ftype,
               PyObject *obj,
               const pvd::BitSet::shared_pointer& bset);

    void store_struct(pvd::PVStructure* fld,
                      const pvd::Structure* ftype,
                      PyObject *obj,
                      const pvd::BitSet::shared_pointer& bset);

    void store_union(pvd::PVUnion* fld,
                      const pvd::Union* ftype,
                      PyObject *obj);

    // assignment of PVStructure from (possibly unrelated) PVStructure

    void store_struct(pvd::PVStructure* fld,
                      pvd::BitSet &changed,
                      const pvd::PVStructure& obj,
                      const pvd::BitSet::shared_pointer &bset);

    void store_union(pvd::PVUnion* fld,
                      const pvd::Union* ftype,
                      const pvd::PVUnion& obj);

    // fetch PVField as Object

    PyObject *fetchfld(pvd::PVField *fld,
                       const pvd::Field *ftype,
                       const pvd::BitSet::shared_pointer& bset,
                       bool unpackstruct,
                       bool unpackrecurse=true,
                       PyObject* wrapper=0);
};

}//namespace

typedef PyClassWrapper<Value> P4PValue;


PyClassWrapper_DEF(P4PValue, "Value")

namespace {

#define TRY P4PValue::reference_type SELF = P4PValue::unwrap(self); try

struct npmap {
    NPY_TYPES npy;
    pvd::ScalarType pvd;
};
const npmap np2pvd[] = {
    {NPY_BOOL, pvd::pvBoolean}, // bool stored as one byte
    {NPY_INT8, pvd::pvByte},
    {NPY_INT16, pvd::pvShort},
    {NPY_INT32, pvd::pvInt},
    {NPY_INT64, pvd::pvLong},
    {NPY_UINT8, pvd::pvUByte},
    {NPY_UINT16, pvd::pvUShort},
    {NPY_UINT32, pvd::pvUInt},
    {NPY_UINT64, pvd::pvULong},
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

// so far not needed...
//pvd::ScalarType ptype(NPY_TYPES t) {
//    for(const npmap *p = np2pvd; p->npy!=NPY_NOTYPE; p++) {
//        if(p->npy==t) return p->pvd;
//    }
//    throw std::runtime_error(SB()<<"Unable to map npy type '"<<(int)t<<"'");
//}


void Value::store_struct(pvd::PVStructure* fld,
                         const pvd::Structure* ftype,
                         PyObject *obj,
                         const pvd::BitSet::shared_pointer& bset)
{
    if(PyDict_Check(obj)) {
        Py_ssize_t n=0;
        PyObject *K, *V;
        while(PyDict_Next(obj, &n, &K, &V)) {
            PyString key(K);
            pvd::PVFieldPtr F(fld->getSubField(key.str()));
            if(!F) {
                PyErr_Format(PyExc_KeyError, "no sub-field %s.%s", fld->getFullName().c_str(), key.str().c_str());
                throw std::runtime_error("not seen");
            }
            storefld(F.get(), F->getField().get(), V, bset);
        }

    } else if(PyObject_IsInstance(obj, (PyObject*)P4PValue_type)) {
        Value& W = P4PValue::unwrap(obj);
        pvd::BitSet changed;
        if(W.I)
            changed = *W.I;
        else
            changed.set(0);
        store_struct(fld, changed, *W.V, bset);

    } else if(ftype->getID()=="enum_t") {
        // Attempted enumeration assignment

        pvd::PVScalar::shared_pointer index(fld->getSubField<pvd::PVScalar>("index"));
        pvd::PVStringArray::const_shared_pointer choices(fld->getSubField<pvd::PVStringArray>("choices"));

        if(!index || !choices)
            throw std::runtime_error("enum_t assignment finds non-complient enum_t");

        if(PyUnicode_Check(obj) || PyBytes_Check(obj)) {
            PyString pystr(obj);
            std::string str(pystr.str());

            pvd::PVStringArray::const_svector C(choices->view());

            if(C.empty())
                PyErr_WarnEx(PyExc_UserWarning, "enum_t assignment with empty choices", 2);

            bool found = false;

            // search for matching choices string
            for(size_t i=0; i<C.size(); i++) {
                if(C[i]==str) {
                    // match
                    index->putFrom<pvd::int32>(i);
                    found = true;
                    break;
                }
            }

            // attempt to convert from string
            if(!found) {
                try {
                    index->putFrom(str);
                } catch(std::runtime_error& e) {
                    PyErr_Format(PyExc_ValueError, "%s", e.what());
                    throw std::runtime_error("not seen");
                }
            }
        } else {
            long val = PyLong_AsLong(obj);
            if(PyErr_Occurred())
                throw std::runtime_error("not seen");

            index->putFrom<pvd::int32>(val);
        }

        if(bset)
            bset->set(index->getFieldOffset());

    } else {
        // an iterable yielding tuples ('fieldname', value)
        PyRef iter;
        try {
            PyRef temp(PyObject_GetIter(obj));
            iter.swap(temp);
        }catch(std::runtime_error&){
            PyErr_Format(PyExc_KeyError, "Can't assign sub-structure \"%s\" from %s : expect iterable/list of tuples",
                         fld->getFullName().c_str(), Py_TYPE(obj)->tp_name);
            throw std::runtime_error("not seen");
        }

        while(true) {
            PyRef I(PyIter_Next(iter.get()), nextiter());
            if(!I) break;

            const char *key = 0;
            PyObject *V = 0;

            if(!PyTuple_Check(I.get())) {
                PyErr_Format(PyExc_ValueError, "Assigned object must be iterable yielding a typle ('name', value)");
                throw std::runtime_error("XXX");
            }

            if(!PyArg_ParseTuple(I.get(), "sO", &key, &V))
                throw std::runtime_error("XXX");

            pvd::PVFieldPtr F(fld->getSubField(key));
            if(!F) {
                PyErr_Format(PyExc_KeyError, "no sub-field %s.%s", fld->getFullName().c_str(), key);
                throw std::runtime_error("not seen");
            }

            storefld(F.get(), F->getField().get(), V, bset);
        }
    }
}

/* depth-first iteration of 'obj'.
 * copy each marked field of 'obj' into a field in 'fld'
 * with the same name, and similar type (or error).
 * un-marked fields in 'obj' are ignored.
 *
 * Neither side is required to be a root (so can't use full name lookup).
 */
void Value::store_struct(pvd::PVStructure* fld,
                         pvd::BitSet& changed, // in obj
                         const pvd::PVStructure& obj,
                         const pvd::BitSet::shared_pointer &bset) // fld
{
    const pvd::StructureConstPtr& stype = obj.getStructure();

    const pvd::StringArray& names = stype->getFieldNames();
    const pvd::FieldConstPtrArray& types = stype->getFields();
    // TODO: getPVFields() breaks const-ness
    const pvd::PVFieldPtrArray& fields = obj.getPVFields();

    const size_t first=obj.getFieldOffset(),
                 next=obj.getNextFieldOffset();

    if(changed.get(first)) {
        // expand compressed
        for(size_t i=first+1, N=next; i<N; i++) {
            changed.set(i);
        }
        changed.clear(first);
    }

    {
        pvd::uint32 firstChanged = changed.nextSetBit(first+1);
        if(firstChanged<0 || firstChanged>=next)
            return; // nothing to copy in this (sub)struct
    }

    for(size_t i=0, N=names.size(); i<N; i++) {
        size_t subfirst = changed.nextSetBit(fields[i]->getFieldOffset());
        if(subfirst<0 || subfirst>=fields[i]->getNextFieldOffset())
            continue;

        pvd::PVFieldPtr dest(fld->getSubField(names[i]));
        if(!dest) {
            PyErr_Format(PyExc_KeyError, "Can't assign non-existant \"%s.%s\"",
                         fld->getFullName().c_str(), names[i].c_str());
            throw std::runtime_error("not seen");
        }

        const pvd::FieldConstPtr& dtype = dest->getField();

        if(types[i]->getType() != dtype->getType()) {
            PyErr_Format(PyExc_KeyError, "Can't assign \"%s.%s\" %s from %s",
                         fld->getFullName().c_str(), names[i].c_str(),
                         pvd::TypeFunc::name(types[i]->getType()),
                         pvd::TypeFunc::name(dtype->getType()));
            throw std::runtime_error("not seen");
        }

        switch(dtype->getType()) {
        case pvd::scalar: {
            pvd::PVScalar* F = static_cast<pvd::PVScalar*>(dest.get());
            pvd::PVScalar* S = static_cast<pvd::PVScalar*>(fields[i].get());
            pvd::AnyScalar temp;
            S->getAs(temp);
            F->putFrom(temp);
            if(bset)
                bset->set(F->getFieldOffset());
        }
            break;
        case pvd::scalarArray: {
            pvd::PVScalarArray* F = static_cast<pvd::PVScalarArray*>(dest.get());
            pvd::PVScalarArray* S = static_cast<pvd::PVScalarArray*>(fields[i].get());
            pvd::shared_vector<const void> temp;
            S->getAs(temp);
            F->putFrom(temp);
            if(bset)
                bset->set(F->getFieldOffset());
        }
            break;
        case pvd::structure: {
            pvd::PVStructure* F = static_cast<pvd::PVStructure*>(dest.get());
            pvd::PVStructure* S = static_cast<pvd::PVStructure*>(fields[i].get());
            store_struct(F, changed, *S, bset);
        }
            break;
        case pvd::structureArray: {
            pvd::PVStructureArray* F = static_cast<pvd::PVStructureArray*>(dest.get());
            pvd::PVStructureArray* S = static_cast<pvd::PVStructureArray*>(fields[i].get());
            pvd::StructureConstPtr Ftype = F->getStructureArray()->getStructure();
            pvd::PVStructureArray::const_svector src(S->view());
            pvd::PVStructureArray::svector dest(src.size());
            const pvd::PVDataCreatePtr& create(pvd::getPVDataCreate());
            pvd::BitSet dummy;

            for(size_t i=0, N=src.size(); i<N; i++) {
                if(!src[i])
                    continue;
                dest[i] = create->createPVStructure(Ftype);
                dummy.clear();
                dummy.set(0);
                store_struct(dest[i].get(), dummy, *src[i], pvd::BitSetPtr());
            }

            F->replace(pvd::freeze(dest));
            if(bset)
                bset->set(F->getFieldOffset());
        }
            break;
        case pvd::union_: {
            pvd::PVUnion* F = static_cast<pvd::PVUnion*>(dest.get());
            pvd::PVUnion* S = static_cast<pvd::PVUnion*>(fields[i].get());
            store_union(F, F->getUnion().get(), *S);
            if(bset)
                bset->set(F->getFieldOffset());
        }
            break;
        case pvd::unionArray: {
            pvd::PVUnionArray* F = static_cast<pvd::PVUnionArray*>(dest.get());
            pvd::PVUnionArray* S = static_cast<pvd::PVUnionArray*>(fields[i].get());
            pvd::UnionConstPtr Ftype = F->getUnionArray()->getUnion();
            pvd::PVUnionArray::const_svector src(S->view());
            pvd::PVUnionArray::svector dest(src.size());
            const pvd::PVDataCreatePtr& create(pvd::getPVDataCreate());

            for(size_t i=0, N=src.size(); i<N; i++) {
                if(!src[i])
                    continue;
                dest[i] = create->createPVUnion(Ftype);
                store_union(dest[i].get(), Ftype.get(), *src[i]);
            }

            F->replace(pvd::freeze(dest));
            if(bset)
                bset->set(F->getFieldOffset());
        }
            break;
        default:
            throw std::runtime_error(SB()<<__FILE__<<":"<<__LINE__<<" Not implemented "<<pvd::TypeFunc::name(dtype->getType()));
        }
    }
}


void Value::store_union(pvd::PVUnion* fld,
                        const pvd::Union* ftype,
                        PyObject *obj)
{
    pvd::PVField::shared_pointer U;

    if(obj==Py_None) {
        // assign any Union w/ None to clear

        // This will fail with pvDataCPP <= 6.0.0
        // due to a bug
#ifdef PVDATA_VERSION_INT
#if PVDATA_VERSION_INT >= VERSION_INT(7, 0, 0, 0)
        fld->set(pvd::PVUnion::UNDEFINED_INDEX, pvd::PVFieldPtr());
#else
        throw std::runtime_error("Clear PVUnion is broken is pvData < 7.0.0");
#endif
#else
        throw std::runtime_error("Clear PVUnion is broken is pvData < 7.0.0");
#endif
        return;

    } else if(ftype->isVariant()) {
        // assign variant with plain value or wrapped Structure

        if(PyObject_TypeCheck(obj, P4PValue_type)) {
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

        U = fld->select(select);

        if(PyObject_TypeCheck(val, P4PValue_type)) {
            pvd::PVStructure::shared_pointer V(P4PValue_unwrap(val));
            if(V->getField().get()==U->getField().get())
                fld->set(V); // store exact
            else if(U->getField()->getType()==pvd::structure)
                std::tr1::static_pointer_cast<pvd::PVStructure>(U)->copy(*V); // copy similar
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
        pvd::BitSet::shared_pointer empty;
        for(size_t i=0, N=ftype->getNumberFields(); i<N; i++) {
            U = fld->select(i);
            try {
                storefld(U.get(),
                         U->getField().get(),
                         obj,
                         empty);
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

    // no tracking inside unions
    pvd::BitSet::shared_pointer empty;

    storefld(U.get(),
             U->getField().get(),
             obj,
             empty);

    fld->set(U);
}

void Value::store_union(pvd::PVUnion* fld,
                        const pvd::Union* ftype,
                        const pvd::PVUnion& src)
{
    const pvd::UnionConstPtr& stype(src.getUnion());
    const pvd::PVField::const_shared_pointer val(src.get()); // may be null

    if(ftype->isVariant()) {
        // if dest is variant, then copy
        if(val) {
            pvd::PVFieldPtr temp(pvd::getPVDataCreate()->createPVField(val->getField()));
            temp->copyUnchecked(*val);
            fld->set(temp);
        } else {
            fld->select(pvd::PVUnion::UNDEFINED_INDEX);
        }

    } else if(!stype->isVariant()) {
        // neither is variant
        // require that selected field names match
        if(src.getSelectedIndex() == pvd::PVUnion::UNDEFINED_INDEX || !val) {
            fld->select(pvd::PVUnion::UNDEFINED_INDEX);
        } else {
            pvd::PVFieldPtr temp(pvd::getPVDataCreate()->createPVField(val->getField()));
            temp->copyUnchecked(*val);
            try {
                fld->set(src.getSelectedFieldName(), temp);
            }catch(std::invalid_argument& e){
                throw std::runtime_error(SB()<<"Error assigning union \""<<fld->getFullName()<<"\" : "<<e.what());
            }
        }

    } else {
        // src is variant, dest is not
        // attempt automatic selection
        if(!val) {
            fld->select(pvd::PVUnion::UNDEFINED_INDEX);
        } else {
            const pvd::FieldConstPtr& vtype(val->getField());
            pvd::ScalarType guess_scalar;
            switch(vtype->getType()) {
            case pvd::scalar:
                guess_scalar = static_cast<const pvd::Scalar&>(*vtype).getScalarType();
                break;
            case pvd::scalarArray:
                guess_scalar = static_cast<const pvd::ScalarArray&>(*vtype).getElementType();
                break;
            default:
                guess_scalar = pvd::pvString;
            }

            fld->select(ftype->guess(vtype->getType(), guess_scalar));

            pvd::PVFieldPtr temp(pvd::getPVDataCreate()->createPVField(val->getField()));
            temp->copyUnchecked(*val);
            fld->set(temp);
        }
    }
}

void Value::storefld(pvd::PVField* fld,
                     const pvd::Field* ftype,
                     PyObject *obj,
                     const pvd::BitSet::shared_pointer& bset)
{
    const size_t fld_offset = fld->getFieldOffset();

    switch(ftype->getType()) {
    case pvd::scalar: {
        pvd::PVScalar* F = static_cast<pvd::PVScalar*>(fld);
        if(PyBool_Check(obj)) {
            F->putFrom<pvd::boolean>(obj==Py_True);
#if PY_MAJOR_VERSION < 3
        } else if(PyInt_Check(obj)) {
            F->putFrom<pvd::uint64>(PyInt_AsLong(obj));
#endif
        } else if(PyLong_Check(obj) || PyArray_IsScalar(obj, Integer)) {
            int oflow = 0;
            long long temp = PyLong_AsLongLongAndOverflow(obj, &oflow);
            if(!PyErr_Occurred()) {
                if(oflow==0) {
                    F->putFrom<pvd::int64>(temp);
                } else if(oflow==1) { // triggered when obj > 0x7fffffffffffffff
                    F->putFrom<pvd::uint64>(PyLong_AsUnsignedLongLong(obj));
                } else {
                    PyErr_Format(PyExc_OverflowError, "long out of range low");
                }
            }
            if(PyErr_Occurred())
                throw std::runtime_error("XXX");
        } else if(PyFloat_Check(obj) || PyArray_IsScalar(obj, Number)) {
            F->putFrom(PyFloat_AsDouble(obj));
        } else if(PyBytes_Check(obj)) {
            std::string S(PyBytes_AS_STRING(obj), PyBytes_GET_SIZE(obj));
            F->putFrom(S);

        } else if(PyUnicode_Check(obj) || PyArray_IsScalar(obj, Unicode)) {
            PyRef S(PyUnicode_AsUTF8String(obj));

            std::string V(PyBytes_AS_STRING(S.get()), PyBytes_GET_SIZE(S.get()));
            F->putFrom(V);

        } else {
            throw std::runtime_error(SB()<<"Can't assign scalar field "<<fld->getFullName()<<" with "<<Py_TYPE(obj)->tp_name);
        }
    }
        if(bset)
            bset->set(fld_offset);
        return;
    case pvd::scalarArray: {
        pvd::PVScalarArray* F = static_cast<pvd::PVScalarArray*>(fld);
        const pvd::ScalarArray *T = static_cast<const pvd::ScalarArray *>(ftype);
        pvd::ScalarType etype = T->getElementType();

        if(etype==pvd::pvString) {
            PyRef iter(PyObject_GetIter(obj));

            pvd::shared_vector<std::string> vec;

            while(1) {
                PyRef I(PyIter_Next(iter.get()), nextiter());
                if(!I) break;

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
            if(bset)
                bset->set(fld_offset);

        } else {
            NPY_TYPES nptype(ntype(etype));

            PyRef V(PyArray_FromAny(obj, PyArray_DescrFromType(nptype), 0, 0,
                                    NPY_CARRAY_RO, NULL));

            if(PyArray_NDIM(V.get())!=1)
                throw std::runtime_error("Only 1-d array can be assigned");

            // TODO: detect reference cycles so we can avoid this copy
            //       Cycles can be created only if we both store and fetch
            //       by reference.
            pvd::shared_vector<void> buf(pvd::ScalarTypeFunc::allocArray(etype, PyArray_DIM(V.get(), 0)));

            memcpy(buf.data(), PyArray_DATA(V.get()), PyArray_NBYTES(V.get()));

            F->putFrom(pvd::freeze(buf));
            if(bset)
                bset->set(fld_offset);
        }
    }
        return;
    case pvd::structure: {
        pvd::PVStructure *F = static_cast<pvd::PVStructure*>(fld);
        const pvd::Structure *T = static_cast<const pvd::Structure*>(ftype);
        store_struct(F, T, obj, bset);
    }
        return;
    case pvd::structureArray: {
        pvd::PVStructureArray* F = static_cast<pvd::PVStructureArray*>(fld);
        const pvd::StructureArray* T = static_cast<const pvd::StructureArray*>(ftype);
        const pvd::Structure* ST = T->getStructure().get();

        const pvd::PVDataCreatePtr& create = pvd::getPVDataCreate();
        pvd::BitSet::shared_pointer junk;
        pvd::PVStructureArray::svector arr;

        PyRef iter(PyObject_GetIter(obj));
        while(1) {

            PyRef I(PyIter_Next(iter.get()), nextiter());
            if(!I) break;

            pvd::PVStructurePtr elem(create->createPVStructure(T->getStructure()));

            store_struct(elem.get(), ST, I.get(), junk);

            arr.push_back(elem);
        }

        F->replace(pvd::freeze(arr));
        if(bset)
            bset->set(fld_offset);

    }
        return;
    case pvd::union_: {
        pvd::PVUnion* F = static_cast<pvd::PVUnion*>(fld);
        const pvd::Union *T = static_cast<const pvd::Union *>(ftype);

        store_union(F, T, obj);
        if(bset)
            bset->set(fld_offset);
    }
        return;
    case pvd::unionArray: {
        pvd::PVUnionArray* F = static_cast<pvd::PVUnionArray*>(fld);
        pvd::UnionConstPtr T = static_cast<const pvd::UnionArray *>(ftype)->getUnion();

        pvd::PVUnionArray::svector arr;
        PyRef iter(PyObject_GetIter(obj));

        const pvd::PVDataCreatePtr& create = pvd::getPVDataCreate();

        while(true) {
            PyRef item(PyIter_Next(iter.get()), nextiter());
            if(!item) break;

            pvd::PVUnionPtr dest(create->createPVUnion(T));
            store_union(dest.get(), T.get(), item.get());

            arr.push_back(dest);
        }

        F->replace(pvd::freeze(arr));
        if(bset)
            bset->set(fld_offset);
    }
        return;
    }

    throw std::runtime_error(SB()<<"Storage of type not implemented : "<<pvd::TypeFunc::name(ftype->getType()));
}

PyObject *Value::fetchfld(pvd::PVField *fld,
                          const pvd::Field *ftype,
                          const pvd::BitSet::shared_pointer& bset,
                          bool unpackstruct,
                          bool unpackrecurse,
                          PyObject *wrapper)
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
#if PY_MAJOR_VERSION < 3
            return PyInt_FromLong(F->getAs<pvd::int32>());
#else
            return PyLong_FromLong(F->getAs<pvd::int32>());
#endif
        case pvd::pvLong:
            return PyLong_FromLongLong(F->getAs<pvd::int64>());
        case pvd::pvUInt:
        case pvd::pvULong:
            return PyLong_FromUnsignedLongLong(F->getAs<pvd::uint64>());
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
                                    esize, NPY_CARRAY_RO, NULL));

            PyObject *base = P4PArray_make(arr);
            ((PyArrayObject*)pyarr.get())->base = base;

            return pyarr.release();
        }
    }
        break;
    case pvd::structure: {
        pvd::PVStructure* F = static_cast<pvd::PVStructure*>(fld);
        const pvd::Structure *T = static_cast<const pvd::Structure*>(ftype);

        if(unpackstruct) {
            const pvd::StringArray& names(T->getFieldNames());
            const pvd::FieldConstPtrArray& flds(T->getFields());
            const pvd::PVFieldPtrArray& vals(F->getPVFields());

            PyRef list(PyList_New(vals.size()));

            for(size_t i=0; i<vals.size(); i++) {
                PyRef val(fetchfld(vals[i].get(), flds[i].get(), bset, unpackrecurse, true, wrapper));

                PyRef item(Py_BuildValue("sO", names[i].c_str(), val.get()));

                PyList_SET_ITEM(list.get(), i, item.release());
            }

            if (wrapper) {
                PyRef dict(PyObject_CallFunction(wrapper, "O", list.get()));
                return dict.release();
            }

            return list.release();

        } else {
            PyObject *self = P4PValue::wrap(this);
            return P4PValue_wrap(Py_TYPE(self), std::tr1::static_pointer_cast<pvd::PVStructure>(F->shared_from_this()), bset);

        }
    }
        break;
    case pvd::structureArray: {
        pvd::PVStructureArray* F = static_cast<pvd::PVStructureArray*>(fld);
        const pvd::StructureArray *T = static_cast<const pvd::StructureArray*>(ftype);
        const pvd::Structure *ST = T->getStructure().get();
        pvd::BitSet::shared_pointer empty;
        pvd::PVStructureArray::const_svector arr(F->view());

        PyRef list(PyList_New(arr.size()));

        for(size_t i=0, N=arr.size(); i<N; i++) {
            if(!arr[i]) {
                Py_INCREF(Py_None);
                PyList_SET_ITEM(list.get(), i, Py_None);

            } else {
                PyObject *elem = fetchfld(arr[i].get(), ST, empty, unpackstruct, unpackrecurse, wrapper);

                PyList_SET_ITEM(list.get(), i, elem);
            }
        }

        return list.release();
    }

        break;
    case pvd::union_: {
        pvd::PVUnion* F = static_cast<pvd::PVUnion*>(fld);
        //const pvd::Union *T = static_cast<const pvd::Union *>(ftype);

        pvd::PVFieldPtr val(F->get());
        if(!val)
            Py_RETURN_NONE;
        else
            return fetchfld(val.get(), val->getField().get(), bset, unpackstruct, true, wrapper);
    }
        break;
    case pvd::unionArray: {
        pvd::PVUnionArray* F = static_cast<pvd::PVUnionArray*>(fld);

        pvd::PVUnionArray::const_svector arr(F->view());

        PyRef list(PyList_New(arr.size()));
        pvd::BitSet::shared_pointer empty;

        for(size_t i=0; i<arr.size(); i++) {
            PyRef ent;
            pvd::PVFieldPtr val;

            if(!arr[i] || !(val=arr[i]->get())) {
                ent.reset(Py_None, borrow());
            } else {
                ent.reset(fetchfld(val.get(), val->getField().get(), empty, unpackstruct, true, wrapper));
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
        PyObject *type = NULL, *value = Py_None;
        PyObject *clone = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO!", (char**)names,
                                        &type,
                                        &value,
                                        P4PValue_type, &clone))
            return -1;

        if(SELF.V) {
            // magic construction w/ P4PValue_wrap()

        } else if(type) {
            pvd::Structure::const_shared_pointer S(P4PType_unwrap(type));

            pvd::PVStructure::shared_pointer V(pvd::getPVDataCreate()->createPVStructure(S));

            SELF.I.reset(new pvd::BitSet(V->getNextFieldOffset()));

            if(value!=Py_None) {
                SELF.store_struct(V.get(), S.get(), value, SELF.I);
            }

            SELF.V = V;

        } else if(clone) {
            const P4PValue::reference_type other = P4PValue::unwrap(clone);
            SELF.V = other.V;
            SELF.I.reset(new pvd::BitSet);
            *SELF.I = *other.I;

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
                       value,
                       SELF.I);

        return 0;
    }CATCH()
    return -1;

}

PyObject* P4PValue_getattr(PyObject *self, PyObject *name)
{
    TRY {
        PyObject *ret = PyObject_GenericGetAttr((PyObject*)self, name);
        if(ret)
            return ret;
        // there is an AttributeError

        PyString S(name);
        pvd::PVFieldPtr fld = SELF.V->getSubField(S.str());
        if(!fld)
            return 0;
        PyErr_Clear(); // clear AttributeError

        // return sub-struct as Value
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             false);
    }CATCH()
    return NULL;
}

PyObject* P4PValue_toList(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        const char *names[] = {"name", NULL};
        const char *name = NULL;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|z", (char**)names, &name))
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

        // return sub-struct as list of tuple
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             true);

    }CATCH()
    return NULL;
}

PyObject* P4PValue_toDict(PyObject *self, PyObject *args, PyObject *kws)
{
    TRY {
        const char *names[] = {"name", "type", NULL};
        const char *name = NULL;
        PyObject *wrapper = (PyObject*)&PyDict_Type;
        if(!PyArg_ParseTupleAndKeywords(args, kws, "|zO!", (char**)names, &name, &PyType_Type, &wrapper))
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

        // return sub-struct as list of tuple
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             true,
                             true,
                             wrapper);

    }CATCH()
    return NULL;
}


PyObject* P4PValue_items(PyObject *self, PyObject *args)
{
    TRY {
        const char *name = NULL;
        if(!PyArg_ParseTuple(args, "|z", &name))
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

        // return sub-struct as list of tuple, not recursive
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             true, false);

    }CATCH()
    return NULL;
}

struct limited_strbuf : public std::streambuf {
    std::vector<std::streambuf::char_type> buf;
    size_t limit;
    explicit limited_strbuf(size_t limit)
        :buf(limit+4, '\0')
        ,limit(limit)
    {
        setp(&buf[0], &buf[limit]);
        buf[limit] = '.';
        buf[limit+1] = '.';
        buf[limit+2] = '.';
        // limit+3 = '\0'
    }
};

PyObject* P4PValue_tostr(PyObject *self, PyObject *args, PyObject *kwds)
{
    TRY {
        const char *names[] = {"limit", NULL};
        unsigned long limit = 0;
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "|k", (char**)names, &limit))
            return NULL;

        if(limit==0) {
            std::ostringstream strm;
            strm<<SELF.V;

            return PyUnicode_FromString(strm.str().c_str());
        } else {
            limited_strbuf buf(limit);
            std::ostream strm(&buf);

            strm<<SELF.V;

            return PyUnicode_FromString(&buf.buf[0]);
        }
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

PyObject *P4PValue_has(PyObject *self, PyObject *args)
{
    TRY {
        const char *name;
        if(!PyArg_ParseTuple(args, "s", &name))
            return NULL;

        if(SELF.V->getSubField(name))
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    }CATCH()
    return NULL;
}

PyObject *P4PValue_get(PyObject *self, PyObject *args)
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

        // return sub-struct as Value
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             false);
    }CATCH()
    return NULL;
}

PyObject *P4PValue_id(PyObject *self)
{
    TRY {
        return PyUnicode_FromString(SELF.V->getStructure()->getID().c_str());
    }CATCH()
    return NULL;
}

PyObject *P4PValue_gettype(PyObject *self, PyObject *args)
{
    TRY {
        const char *name = NULL;
        if(!PyArg_ParseTuple(args, "|z", &name))
            return NULL;
        pvd::StructureConstPtr T;
        if(!name) {
            T = SELF.V->getStructure();
        } else {
            pvd::PVFieldPtr F(SELF.V->getSubField(name));
            if(!F)
                return PyErr_Format(PyExc_KeyError, "No field %s", name);
            pvd::FieldConstPtr FT(F->getField());
            if(FT->getType()==pvd::structure) {
                T = std::tr1::static_pointer_cast<const pvd::Structure>(FT);
            } else {
                return PyErr_Format(PyExc_KeyError, "Can't extract type of non-struct field %s", name);
            }
        }
        return P4PType_wrap(P4PType_type, T);
    }CATCH()
    return NULL;
}

PyObject* P4PValue_changed(PyObject *self, PyObject *args, PyObject *kws)
{
    static const char* names[] = {"field", NULL};
    const char* fname = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, kws, "|z", (char**)names, &fname))
        return NULL;
    TRY {

        if(!SELF.I)
            Py_RETURN_TRUE;

        pvd::PVField::shared_pointer fld;
        if(fname)
            fld = SELF.V->getSubField(fname);
        else
            fld = SELF.V;
        if(!fld)
            return PyErr_Format(PyExc_KeyError, "%s", fname);

        // is the bit associated with this field set?
        const size_t offset = fld->getFieldOffset();
        if(SELF.I->get(offset))
            Py_RETURN_TRUE;

        // are any parent bits set?
        for(pvd::PVStructure *parent = fld->getParent(); parent; parent = parent->getParent())
        {
            if(SELF.I->get(parent->getFieldOffset()))
                Py_RETURN_TRUE;
        }

        // are any child bits set?
        const size_t nextoffset = fld->getNextFieldOffset();
        const size_t nextset = SELF.I->nextSetBit(offset+1);

        if(nextset>offset && nextset<nextoffset)
            Py_RETURN_TRUE;

        Py_RETURN_FALSE;
    }CATCH()
    return NULL;
}

PyObject* P4PValue_mark(PyObject *self, PyObject *args, PyObject *kws)
{
    static const char* names[] = {"field", "val", NULL};
    const char* fname = NULL;
    PyObject *val = Py_True;
    if(!PyArg_ParseTupleAndKeywords(args, kws, "|zO", (char**)names, &fname, &val))
        return NULL;
    TRY {
        bool B = PyObject_IsTrue(val);

        if(SELF.I) {
            pvd::PVField::shared_pointer fld;
            if(fname)
                fld = SELF.V->getSubField(fname);
            else
                fld = SELF.V;
            if(!fld)
                return PyErr_Format(PyExc_KeyError, "%s", fname);

            SELF.I->set(fld->getFieldOffset(), B);

            //TODO: how to handle when parent bits are set???
        } else {
            //TODO: lazy create bitset?
        }

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject* P4PValue_unmark(PyObject *self)
{
    TRY {
        if(SELF.I)
            SELF.I->clear();

        Py_RETURN_NONE;
    }CATCH()
    return NULL;
}

PyObject* P4PValue_changedSet(PyObject *self, PyObject *args, PyObject *kws)
{
    static const char* names[] = {"expand", "parents", NULL};
    PyObject *pyexpand = Py_False, *pyparents = Py_False;
    if(!PyArg_ParseTupleAndKeywords(args, kws, "|OO", (char**)names, &pyexpand, &pyparents))
        return NULL;
    TRY {
        // use of PVField::getFullName() prefixes with names of parents,
        // which can't be used to index of this sub-structure.
        if(SELF.V->getParent())
            return PyErr_Format(PyExc_NotImplementedError, "changedSet not implemented for sub-struct");

        size_t b0 = SELF.V->getFieldOffset(),
               b1 = SELF.V->getNextFieldOffset();

        bool expand = PyObject_IsTrue(pyexpand);
        bool parents = PyObject_IsTrue(pyparents);

        pvd::BitSet changed;
        if(SELF.I && !SELF.I->get(0)) {
            changed = *SELF.I;
        } else {
            // no tracking, or all changed
            for(size_t i=b0+1; i<b1; i++)
                changed.set(i);
        }

        PyRef ret(PySet_New(NULL));

        for(pvd::int32 i=changed.nextSetBit(b0+1); i>=0 && size_t(i)<b1; i=changed.nextSetBit(i+1)) {
            const pvd::PVField& subfld = *SELF.V->getSubFieldT(i);
            if(!expand || subfld.getField()->getType()!=pvd::structure) {
                PyRef N(PyUnicode_FromString(subfld.getFullName().c_str()));
                if(PySet_Add(ret.get(), N.get()))
                    return NULL;

            } else {
                for(size_t j=i+1, J=subfld.getNextFieldOffset(); j<J; j++) {
                    changed.set(j);
                }
            }
            if(parents) {
                // all parents except the root
                for(const pvd::PVStructure *parent = subfld.getParent(); parent && parent->getParent(); parent=parent->getParent()) {
                    PyRef N(PyUnicode_FromString(parent->getFullName().c_str()));
                    if(PySet_Add(ret.get(), N.get()))
                        return NULL;
                }
            }
        }

        return ret.release();
    }CATCH()
    return NULL;
}

PyObject* P4PValue_magic(PyObject *self, PyObject *args)
{
    try {
        PyObject *replacement, *old;
        if(!PyArg_ParseTuple(args, "O", &replacement))
            return NULL;

        if(!PyObject_IsSubclass(replacement, (PyObject*)&P4PValue::type))
            return PyErr_Format(PyExc_ValueError, "Not sub-class");

        old = (PyObject*)P4PValue_type;
        P4PValue_type = (PyTypeObject*)replacement;

        Py_INCREF(replacement);
        Py_DECREF(old);

        Py_RETURN_NONE;
    } CATCH()
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
        pvd::PVFieldPtr fld;
        if(name == Py_None) {
            fld = SELF.V;
            assert(!!fld);

        } else {
            PyString S(name);
            fld = SELF.V->getSubField(S.str());
            if(!fld) {
                PyErr_SetString(PyExc_KeyError, S.str().c_str());
                return -1;
            }
        }

        SELF.storefld(fld.get(),
                       fld->getField().get(),
                       value,
                       SELF.I);

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

        // return sub-struct as Value
        return SELF.fetchfld(fld.get(),
                             fld->getField().get(),
                             SELF.I,
                             false);
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
     "tolist(name=None)\n\n"
     "Recursively transform into a list of tuples."},
     {"todict", (PyCFunction)&P4PValue_toDict, METH_VARARGS|METH_KEYWORDS,
      "todict(name=None, type=dict)\n\n"
      "Recursively transform into a dictionary (or other type constructable from a list of tuples)."},
    {"items", (PyCFunction)&P4PValue_items, METH_VARARGS,
     "items( [\"fld\"] )\n\n"
     "Transform into a list of tuples.  Not recursive"},
    {"select", (PyCFunction)&P4PValue_select, METH_VARARGS|METH_KEYWORDS,
     "select(\"fld\", \"member\")\n"
     "pre-select/clear Union"},
    {"has", (PyCFunction)&P4PValue_has, METH_VARARGS,
     "has(\"fld\")\n"
     "Test for existance of field"},
    {"get", (PyCFunction)&P4PValue_get, METH_VARARGS,
     "get(\"fld\", [default])\n"
     "Fetch a field value, or a default if it does not exist"},
    {"getID", (PyCFunction)&P4PValue_id, METH_NOARGS,
     "getID()\n"
     "Return Structure ID string"},
    {"type", (PyCFunction)&P4PValue_gettype, METH_VARARGS,
     "type( [\"fld\"] )\n"
     "\n"
     ":param field str: None or the name of a sub-structure\n"
     ":returns: The :class:`~p4p.Type` describing this Value."},
    // bitset
    {"changed", (PyCFunction)&P4PValue_changed, METH_VARARGS|METH_KEYWORDS,
     "changed(field) -> bool\n\n"
     "Test if field are marked as changed."},
    {"mark", (PyCFunction)&P4PValue_mark, METH_VARARGS|METH_KEYWORDS,
     "mark(\"fld\", val=True)\n\n"
     "set/clear field as changed"},
    {"unmark", (PyCFunction)&P4PValue_unmark, METH_NOARGS,
     "unmark()\n\n"
     "clear all field changed flag."},
    {"changedSet", (PyCFunction)&P4PValue_changedSet, METH_VARARGS|METH_KEYWORDS,
     "changedSet(expand=False) -> set(['...'])\n\n"},
    {"tostr", (PyCFunction)&P4PValue_tostr, METH_VARARGS|METH_KEYWORDS,
     "tostr(limit=0) -> str\n"
     "Return a string representation of the Value.  If limit!=0, output is truncated after ~this many charactors."},
    {"_magic", (PyCFunction)P4PValue_magic, METH_VARARGS|METH_STATIC,
     "Don't call this!"},
    {NULL}
};

const char value_doc[] =     "Value(type, value=None)\n"
        "\n"
        "Structured value container. Supports dict-list and object-list access\n"
        "\n"
        ":param Type type: A :py:class:`Type` describing the structure\n"
        ":param dict value: Initial values to populate the Value\n"
        ;


} // namespace

PyTypeObject* P4PValue_type = &P4PValue::type;

void p4p_value_register(PyObject *mod)
{
    P4PValue::buildType();
    P4PValue::type.tp_doc = value_doc;
    P4PValue::type.tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE;
    P4PValue::type.tp_init = &P4PValue_init;
    P4PValue::type.tp_getattro = &P4PValue_getattr;
    P4PValue::type.tp_setattro = &P4PValue_setattr;

    P4PValue::type.tp_as_mapping = &P4PValue_mapping;

    P4PValue::type.tp_methods = P4PValue_methods;

    P4PValue::finishType(mod, "ValueBase");
}

epics::pvData::PVStructure::shared_pointer P4PValue_unwrap(PyObject *obj,
                                                           epics::pvData::BitSet *set)
{
    if(!PyObject_TypeCheck(obj, &P4PValue::type))
        throw std::runtime_error("Not a _p4p.ValueBase");
    Value& val = P4PValue::unwrap(obj);
    if(set && val.I)
        *set = *val.I;
    return val.V;
}

std::tr1::shared_ptr<epics::pvData::BitSet> P4PValue_unwrap_bitset(PyObject *obj)
{
    if(!PyObject_TypeCheck(obj, &P4PValue::type))
        throw std::runtime_error("Not a _p4p.ValueBase");
    return P4PValue::unwrap(obj).I;
}

PyObject *P4PValue_wrap(PyTypeObject *type,
                        const epics::pvData::PVStructure::shared_pointer& V,
                        const epics::pvData::BitSet::shared_pointer & I)
{
    assert(V);
    if(!PyType_IsSubtype(type, &P4PValue::type))
        throw std::runtime_error("Not a sub-class of _p4p.ValueBase");

    // magic construction of potentially derived type...

    PyRef args(PyTuple_New(0));
    PyRef kws(PyDict_New());

    PyRef ret(type->tp_new(type, args.get(), kws.get()));

    // inject value *before* __init__ of base or derived type runs
    {
        Value& val = P4PValue::unwrap(ret.get());
        val.V = V;
        val.I = I;
    }

    if(type->tp_init(ret.get(), args.get(), kws.get()))
        throw std::runtime_error("XXX");

    return ret.release();
}
