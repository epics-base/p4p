
#include <sstream>

#include <p4p.h>
#include <_p4p.h>

#define NO_IMPORT_ARRAY
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace p4p {

int except_map()
{
    try {
        if(PyErr_Occurred())
            return 0;
        throw;
    }catch(NoConvert& e){
        PyErr_SetString(PyExc_ValueError, e.what());
    }catch(LookupError& e){
        PyErr_SetString(PyExc_KeyError, e.what());
    }catch(std::invalid_argument& e){
        PyErr_SetString(PyExc_TypeError, e.what());
    }catch(std::exception& e){
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return 0;
}

PyObject* asPy(const Value& v, bool unpackstruct, bool unpackrecurse, PyObject* wrapper)
{
    if(!v)
        Py_RETURN_NONE;

    // handle scalars

    switch(v.storageType()) {
    case StoreType::Bool:
        if(v.as<bool>())
            Py_RETURN_TRUE;
        else
            Py_RETURN_FALSE;
    case StoreType::Real:
        return PyFloat_FromDouble(v.as<double>());
    case StoreType::Integer:
#if PY_MAJOR_VERSION < 3
        if(v.type()!=TypeCode::Int64) {
            return PyInt_FromLong(v.as<int32_t>());
        }
#endif
        return PyLong_FromLongLong(v.as<int64_t>());
    case StoreType::UInteger:
        return PyLong_FromUnsignedLongLong(v.as<uint64_t>());
    case StoreType::String:
        return PyUnicode_FromString(v.as<std::string>().c_str());
    case StoreType::Compound:
    case StoreType::Array:
    case StoreType::Null:
        break;
    }

    // handle compound (non-array)

    if(v.type()==TypeCode::Struct) {
        if(!unpackstruct) {
            return pvxs_pack(v);

        } else {
            PyRef mems(PyList_New(0));

            for(auto mem : v.ichildren()) {
                PyRef mval(asPy(mem, unpackrecurse, true, wrapper));
                PyRef tup(Py_BuildValue("sO", v.nameOf(mem).c_str(), mval.obj));

                if(PyList_Append(mems.obj, tup.obj))
                    throw std::runtime_error("XXX");
            }

            if(!wrapper || wrapper==Py_None)
                return mems.release();
            else
                return PyObject_CallFunction(wrapper, "O", mems.obj);
        }

    } else if(v.type()==TypeCode::Union || v.type()==TypeCode::Any) {
        return asPy(v.as<const Value>(), unpackrecurse, true, wrapper);
    }

    auto varr = v.as<shared_array<const void>>();

    // handle compound array

    if(varr.original_type()==ArrayType::Value) { // StructA, UnionA, AnyA
        auto arr = varr.castTo<const Value>();

        PyRef mems(PyList_New(arr.size()));

        for(size_t i=0, N=arr.size(); i<N; i++) {
            auto mem = asPy(arr[i], unpackrecurse, true, wrapper);
            if(!mem)
                return nullptr;

            PyList_SET_ITEM(mems.obj, i, mem);
        }

        return mems.release();

    } else if(varr.original_type()==ArrayType::String) { // StringA
        auto arr = varr.castTo<const std::string>();

        PyRef mems(PyList_New(arr.size()));

        for(size_t i=0, N=arr.size(); i<N; i++) {
            auto mem = PyUnicode_FromString(arr[i].c_str());
            if(!mem)
                return nullptr;

            PyList_SET_ITEM(mems.obj, i, mem);
        }

        return mems.release();
    }

    // handle array of (simple) scalar

    PyRef holder(pvxs_wrap_array(varr));

    NPY_TYPES ntype;
    switch(varr.original_type()) {
#define CASE(NTYPE, PTYPE) case ArrayType::PTYPE: ntype = (NTYPE); break
    CASE(NPY_BOOL,   Bool);
    CASE(NPY_INT8,   Int8);
    CASE(NPY_INT16,  Int16);
    CASE(NPY_INT32,  Int32);
    CASE(NPY_INT64,  Int64);
    CASE(NPY_UINT8,  UInt8);
    CASE(NPY_UINT16, UInt16);
    CASE(NPY_UINT32, UInt32);
    CASE(NPY_UINT64, UInt64);
    CASE(NPY_FLOAT,  Float32);
    CASE(NPY_DOUBLE, Float64);
#undef CASE
    case ArrayType::Null:
        if(v.type().scalarOf().kind()==Kind::Compound || v.type()==TypeCode::StringA)
            return PyList_New(0u);
        else
            Py_RETURN_NONE;
    default:
        throw std::logic_error(SB()<<"logic error in array Value unpack for "<<varr.original_type());
    }

    auto esize = elementSize(varr.original_type());
    npy_intp shape = varr.size();

    PyRef pyarr(PyArray_New(&PyArray_Type, 1, &shape, ntype, nullptr,
                            const_cast<void*>(varr.data()), // should not actually be modifiable
                            esize, NPY_CARRAY_RO, nullptr));

#ifdef PyArray_SetBaseObject
    PyArray_SetBaseObject((PyArrayObject*)pyarr.obj, holder.release());
#else
    ((PyArrayObject*)pyarr.obj)->base = holder.release();
#endif

    return pyarr.release();
}

static
Value inferPy(PyObject* py)
{
    Value v;

    if(pvxs_isValue(py)) {
        v = pvxs_extract(py);

    } else if(PyTuple_Check(py)) { // explicit
        const char* spec;
        PyObject* val;
        if(!PyArg_ParseTuple(py, "sO;Assign Any w/ (code, val).", &spec, &val))
            throw std::runtime_error("XXX");

        bool isarr = spec[0]=='a';
        TypeCode code;
        switch(isarr ? spec[1] : spec[0]) {
#define CASE(C, TYPE) case C: code = TypeCode::TYPE; break
        CASE('?', Bool);
        CASE('s', String);
        CASE('b', Int8);
        CASE('h', Int16);
        CASE('i', Int32);
        CASE('l', Int64);
        CASE('B', UInt8);
        CASE('H', UInt16);
        CASE('I', UInt32);
        CASE('L', UInt64);
        CASE('f', Float32);
        CASE('d', Float64);
#undef CASE
        default:
            throw std::runtime_error(SB()<<"Invalid Any type spec \""<<spec<<"\"");
        }
        if(isarr)
            code = code.arrayOf();

        auto fld(TypeDef(code).create());

        try {
            storePy(fld, val);
        }catch(std::exception& e){
            throw std::logic_error(SB()<<"Error assigning explicit type "<<code<<" with "<<Py_TYPE(py)->tp_name<<" : "<<e.what());
        }

        v = fld;

    } else {
        // automagically infer TypeCode based on python object type

        TypeCode code = TypeCode::Null;
        if(PyBool_Check(py))
            code = TypeCode::Bool;
#if PY_MAJOR_VERSION < 3
        else if(PyInt_Check(py))
            code = TypeCode::Int32;
#endif
        else if(PyLong_Check(py))
            code = TypeCode::Int64;
        else if(PyFloat_Check(py))
            code = TypeCode::Float64;
        else if(PyBytes_Check(py) || PyUnicode_Check(py))
            code = TypeCode::String;
        else if(PyList_Check(py))
            code = TypeCode::StringA;
        else if(PyArray_Check(py)) {
            switch(PyArray_TYPE(py)) {
#define CASE(NTYPE, PTYPE) case NTYPE: code = TypeCode::PTYPE; break
            CASE(NPY_BOOL, BoolA); // bool stored as one byte
            CASE(NPY_INT8, Int8A);
            CASE(NPY_INT16, Int16A);
            CASE(NPY_INT32, Int32A);
            CASE(NPY_INT64, Int64A);
            CASE(NPY_UINT8, UInt8A);
            CASE(NPY_UINT16, UInt16A);
            CASE(NPY_UINT32, UInt32A);
            CASE(NPY_UINT64, UInt64A);
            CASE(NPY_FLOAT, Float32A);
            CASE(NPY_DOUBLE, Float64A);
            CASE(NPY_STRING, StringA);
#undef CASE
            }
        }

        if(code==TypeCode::Null)
            throw std::runtime_error(SB()<<"Unable to infer TypeCode for "<<Py_TYPE(py)->tp_name);

        auto val(TypeDef(code).create());

        try {
            storePy(val, py);
        }catch(std::exception& e){
            throw std::logic_error(SB()<<"Error assigning inferred type "<<code<<" with "<<Py_TYPE(py)->tp_name<<" : "<<e.what());
        }
        v = val;
    }

    return v;
}

void storePy(Value& v, PyObject* py)
{
    if(!v)
        throw std::invalid_argument("Can't assign value to empty field");

    if(pvxs_isValue(py)) {
        v.assign(pvxs_extract(py));
        return;
    }

    // handle scalar

    switch(v.storageType()) {
    case StoreType::Bool:
    case StoreType::Real:
    case StoreType::Integer:
    case StoreType::UInteger:
    case StoreType::String:
        if(PyBool_Check(py)) {
            v.from<bool>(py==Py_True);

#if PY_MAJOR_VERSION < 3
        } else if(PyInt_Check(py)) {
            v.from<int64_t>(PyInt_AsLong(py));
#endif
        } else if(PyLong_Check(py) || PyArray_IsScalar(py, Integer)) {
            int oflow = 0;
            long long temp = PyLong_AsLongLongAndOverflow(py, &oflow);

            if(!PyErr_Occurred()) {
                if(oflow==0) {
                    v.from<int64_t>(temp);

                } else if(oflow==1) { // triggered when obj > 0x7fffffffffffffff
                    v.from<uint64_t>(PyLong_AsUnsignedLongLong(py));

                } else {
                    PyErr_Format(PyExc_OverflowError, "long out of range low");
                }
            }
            if(PyErr_Occurred())
                throw std::runtime_error("XXX");

        } else if(PyFloat_Check(py) || PyArray_IsScalar(py, Number)) {
            v.from(PyFloat_AsDouble(py));

        } else if(PyBytes_Check(py)) {
            std::string S(PyBytes_AS_STRING(py), PyBytes_GET_SIZE(py));
            v.from(S);

        } else if(PyUnicode_Check(py) || PyArray_IsScalar(py, Unicode)) {
            PyRef S(PyUnicode_AsUTF8String(py));

            std::string V(PyBytes_AS_STRING(S.obj), PyBytes_GET_SIZE(S.obj));
            v.from(V);

        } else {
            throw std::invalid_argument(SB()<<"Can't assign "<<v.type()<<" (scalar) field with "<<Py_TYPE(py)->tp_name);
        }
        return;

    case StoreType::Null:
        if(v.type()==TypeCode::Struct) {
            if(v.id()=="enum_t" && !PyDict_Check(py)) {
                // Attempted magic enumeration assignment

                auto index = v.lookup("index");
                auto choices = v.lookup("choices").as<shared_array<const std::string>>();

                if(choices.empty())
                    PyErr_WarnEx(PyExc_UserWarning, "enum_t assignment with empty choices", 2);

                bool found = false;
                if(PyUnicode_Check(py) || PyBytes_Check(py)) {
                    PyRef temp;
                    const char* val;
                    if(PyUnicode_Check(py)) {
                        temp.reset(PyUnicode_AsUTF8String(py));
                        val = PyBytes_AsString(temp.obj);
                    } else {
                        val = PyBytes_AsString(py);
                    }

                    // attempt choice lookup
                    for(size_t i=0, N=choices.size(); i<N; i++) {
                        if(choices[i]==val) {
                            index.from<uint64_t>(i);
                            found = true;
                            break;
                        }
                    }
                }

                // attempt assignment from index (as integer or string)
                if(!found) {
                    storePy(index, py);
                }
                return;

            } else {
                PyRef items;

                if(PyDict_Check(py)) {
                    items.reset(PyDict_Items(py));
                    py = items.obj;
                }

                PyRef iter(PyObject_GetIter(py));

                while(auto item = PyRef::iternext(iter)) {
                    const char* key;
                    PyObject* val;

                    if(!PyArg_ParseTuple(item.obj, "sO;expect list of ('name', value) tuples", &key, &val))
                        throw std::runtime_error("XXX");

                    auto sub(v.lookup(key));
                    storePy(sub, val);
                }
                return;

            }
        }
        break;

    case StoreType::Compound:
        if(py==Py_None) {
            v = unselect;
            return;
        }
        if(v.type()==TypeCode::Union) {
            if(PyTuple_Check(py)) { // select and assign
                const char* select;
                PyObject* val;
                if(!PyArg_ParseTuple(py, "sO;Assign Union w/ (str, val).", &select, &val))
                    throw std::runtime_error("XXX");

                auto mem(v.lookup(std::string("->")+select));
                storePy(mem, val);
                return;

            } else if(auto sel = v.as<Value>()) { // assign to previously selected
                storePy(sel, py);
                return;

            } else {
                // attempt "magic" selection.  (aka try each field until assignment succeeds...)

                for(auto fld : v.iall()) {
                    if(PyErr_Occurred())
                        PyErr_Clear();

                    // note that fld may be temporary storage
                    try {
                        storePy(fld, py);
                    }catch(NoConvert&){
                        continue;
                    }catch(std::logic_error&){
                        continue;
                    }catch(std::exception& e){
                        throw std::logic_error(SB()<<"Unexpected error while attempting to assign "<<v.nameOf(fld)<<" with "<<Py_TYPE(py)->tp_name);
                    }

                    auto name(v.nameOf(fld));
                    v["->"+name].assign(fld);
                    return;
                }

                PyErr_Format(PyExc_ValueError, "Unable to infer Union field to assign a %s", Py_TYPE(py)->tp_name);
                throw std::runtime_error("XXX");
            }

        } else if(v.type()==TypeCode::Any) {
            v.assign(inferPy(py));
            return;
        }
        break;

    case StoreType::Array: {
        if(v.type()==TypeCode::StructA || v.type()==TypeCode::UnionA) {

            if(!PySequence_Check(py))
                throw std::runtime_error(SB()<<"Must assign sequence to struct/union array, not "<<Py_TYPE(py)->tp_name);

            shared_array<Value> arr(PySequence_Size(py));

            for(size_t i=0; i<arr.size(); i++) {
                PyRef elem(PySequence_GetItem(py, i));

                arr[i] = v.allocMember();

                storePy(arr[i], elem.obj);
            }

            v = arr.freeze();
            return;

        } else if(v.type()==TypeCode::AnyA) {

            if(!PySequence_Check(py))
                throw std::runtime_error(SB()<<"Must assign sequence to variant union (any) array, not "<<Py_TYPE(py)->tp_name);

            shared_array<Value> arr(PySequence_Size(py));

            for(size_t i=0; i<arr.size(); i++) {
                PyRef elem(PySequence_GetItem(py, i));

                arr[i] = inferPy(elem.obj);
            }

            v = arr.freeze();
            return;

        } else if(v.type()==TypeCode::StringA) {

            if(!PySequence_Check(py))
                throw std::runtime_error(SB()<<"Must assign sequence to string array, not "<<Py_TYPE(py)->tp_name);

            shared_array<std::string> arr(PySequence_Size(py));

            for(size_t i=0; i<arr.size(); i++) {
                PyRef elem(PySequence_GetItem(py, i));

                if(PyBytes_Check(elem.obj)) {
                    arr[i] = std::string(PyBytes_AS_STRING(elem.obj), PyBytes_GET_SIZE(elem.obj));

                } else if(PyUnicode_Check(elem.obj)) {
                    PyRef B(PyUnicode_AsUTF8String(elem.obj));

                    arr[i] = std::string(PyBytes_AS_STRING(B.obj), PyBytes_GET_SIZE(B.obj));

                } else {
                    throw std::runtime_error(SB()<<"Assign StringA from sequence of str|bytes, not "<<Py_TYPE(elem.obj)->tp_name);
                }
            }

            v = arr.freeze();
            return;

        } else {
            NPY_TYPES ntype;
            switch(v.type().code) {
        #define CASE(NTYPE, PTYPE) case TypeCode::PTYPE ## A: ntype = (NTYPE); break
            CASE(NPY_BOOL,   Bool);
            CASE(NPY_INT8,   Int8);
            CASE(NPY_INT16,  Int16);
            CASE(NPY_INT32,  Int32);
            CASE(NPY_INT64,  Int64);
            CASE(NPY_UINT8,  UInt8);
            CASE(NPY_UINT16, UInt16);
            CASE(NPY_UINT32, UInt32);
            CASE(NPY_UINT64, UInt64);
            CASE(NPY_FLOAT,  Float32);
            CASE(NPY_DOUBLE, Float64);
        #undef CASE
            default:
                throw std::logic_error(SB()<<"logic error in array Value assignment for "<<v.type());
            }

            PyRef arr(PyArray_FromAny(py, PyArray_DescrFromType(ntype), 0, 0, NPY_CARRAY_RO, NULL));

            if(PyArray_NDIM(arr.obj)!=1)
                throw std::logic_error("Only 1-d array can be assigned");

            auto dest(allocArray(v.type().arrayType(), PyArray_DIM(arr.obj, 0)));

            memcpy(dest.data(), PyArray_DATA(arr.obj), PyArray_NBYTES(arr.obj));

            v = dest.freeze();
            return;
        }
    }
        break;
    default:
        break;
    }

    throw std::invalid_argument(SB()<<"Can't assign "<<v.type()<<" field from "<<Py_TYPE(py)->tp_name);
}

namespace {


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

} // namespace

PyObject* tostr(const Value& v, size_t limit, bool showval)
{
    if(limit==0) {
        std::ostringstream strm;
        strm<<v.format().showValue(showval);

        return PyUnicode_FromString(strm.str().c_str());
    } else {
        limited_strbuf buf(limit);
        std::ostream strm(&buf);

        strm<<v;

        return PyUnicode_FromString(&buf.buf[0]);
    }
}

} // namespace p4p
