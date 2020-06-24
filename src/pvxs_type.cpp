
#include <vector>

#include <p4p.h>
#include <_p4p.h>

namespace p4p {

static
Member plainMember(const char* key, const char* spec)
{
    const auto orig = spec;
    bool isarr = spec[0]=='a';
    if(isarr) {
        spec++; // skip 'a'
    }

    TypeCode code;
    switch(spec[0]) {
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
    CASE('v', Any);
#undef CASE
    default:
        throw std::runtime_error(SB()<<"Invalid plain type spec \""<<orig<<"\"");
    }
    if(isarr)
        code = code.arrayOf();
    return Member(code, key);
}

/** Translate python type spec
 *
 * [
 *   ('field', 'typename'),               # simple types
 *   ('field', ('S', 'name'|None, [...])  # sub-struct
 *   ('field', ('U', None, [...])),       # union
 *   ('field', ('aS', 'name'|None, [...]) # sub-struct array
 *   ('field', ('aU', None, [...])),      # union array
 *   ('field', Type)                      # pre-defined sub-struct
 * ]
 */
static
void appendMembers(std::vector<Member>& members, PyObject* spec)
{
    PyRef iter(PyObject_GetIter(spec));

    while(auto item = PyRef::iternext(iter)) {
        const char *key;
        PyObject *val;

        if(!PyArg_ParseTuple(item.obj, "sO;Expected list of tuples eg. (str, object) or (str, ('S', str, object))", &key, &val))
            throw std::runtime_error("XXX");

        //
        if(pvxs_isType(val)) {
            auto prototype = pvxs_extract(val);

            members.push_back(TypeDef(prototype).as(key));

#if PY_MAJOR_VERSION < 3
        } else if(PyBytes_Check(val)) {
            const char *spec = PyBytes_AsString(val);
            members.push_back(plainMember(key, spec));
#endif
        } else if(PyUnicode_Check(val)) {
            PyRef str(PyUnicode_AsASCIIString(val));
            const char *spec = PyBytes_AsString(str.obj);
            members.push_back(plainMember(key, spec));

        } else {
            const char *mspec, *mid;
            PyObject *mdefs;

            if(!PyArg_ParseTuple(val, "szO;Expected list of tuples (str, str, object) or (str, object)", &mspec, &mid, &mdefs))
                throw std::runtime_error("XXX");

            const auto morig = mspec;
            TypeCode mcode;
            bool isarr = mspec[0]=='a';
            if(isarr)
                mspec++; // skip 'a'

            if(mspec[0]=='s' || mspec[0]=='S') {
                mcode = TypeCode::Struct;
            } else if(mspec[0]=='u' || mspec[0]=='U') {
                mcode = TypeCode::Union;
            } else {
                throw std::runtime_error(SB()<<"Invalid complex type spec \""<<morig<<"\"");
            }

            if(isarr)
                mcode = mcode.arrayOf();

            std::vector<Member> mmembers;
            appendMembers(mmembers, mdefs);

            members.push_back(Member(mcode, key, mid ? mid : "", mmembers));
        }
    }
}

TypeDef startPrototype(const std::string& id, const Value& base)
{
    if(base) {
        return TypeDef(base);
    } else {
        return TypeDef(TypeCode::Struct, id, {});
    }
}

void appendPrototype(TypeDef& def, PyObject* spec)
{
    std::vector<Member> members;
    appendMembers(members, spec);
    def += members;
}

PyObject* asPySpec(const Value& v, bool fakearray)
{
    if(!v)
        throw std::runtime_error("No spec for empty field");

    char spec[3] = "\0\0";
    switch(v.type().scalarOf().code) {
#define CASE(C, TYPE) case TypeCode::TYPE: spec[0] = C; break
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
    CASE('v', Any);
    CASE('S', Struct);
    CASE('U', Union);
#undef CASE
    default:
        throw std::logic_error(SB()<<"Unknown/corrupt type "<<v.type());
    }

    if(v.type().isarray() || fakearray) {
        spec[1] = spec[0];
        spec[0] = 'a';
    }

    if(v.type()==TypeCode::Struct || v.type()==TypeCode::Union) {

        PyRef members(PyList_New(0));

        for(auto mem : v.ichildren()) {
            PyRef mspec(asPySpec(mem));
            PyRef tup(Py_BuildValue("sO", v.nameOf(mem).c_str(), mspec.obj));

            if(PyList_Append(members.obj, tup.obj))
                throw std::runtime_error("XXX");
        }

        return Py_BuildValue("szO", spec,
                             v.id().empty() ? "structure" : v.id().c_str(),
                             members.obj);

    } else if(v.type()==TypeCode::StructA || v.type()==TypeCode::UnionA) {
        // unpack Struct/Union
        // TODO: avoid allocating two temporaries?
        return asPySpec(v.cloneEmpty().allocMember(), true);

    } else {
        return Py_BuildValue("s", spec);
    }
}

} // namespace p4p
