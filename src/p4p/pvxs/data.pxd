#cython: language_level=2

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "<pvxs/data.h>" namespace "pvxs" nogil:
    cdef enum code_t "pvxs::TypeCode::code_t":
        Null "pvxs::TypeCode::Null"

        Bool "pvxs::TypeCode::Bool"
        Int8 "pvxs::TypeCode::Int8"
        Int16 "pvxs::TypeCode::Int16"
        Int32 "pvxs::TypeCode::Int32"
        Int64 "pvxs::TypeCode::Int64"
        UInt8 "pvxs::TypeCode::UInt8"
        UInt16 "pvxs::TypeCode::UInt16"
        UInt32 "pvxs::TypeCode::UInt32"
        UInt64 "pvxs::TypeCode::UInt64"
        Float32 "pvxs::TypeCode::Float32"
        Float64 "pvxs::TypeCode::Float64"
        String "pvxs::TypeCode::String"
        Struct "pvxs::TypeCode::Struct"
        Union "pvxs::TypeCode::Union"
        Any "pvxs::TypeCode::Any"

        BoolA "pvxs::TypeCode::BoolA"
        Int8A "pvxs::TypeCode::Int8A"
        Int16A "pvxs::TypeCode::Int16A"
        Int32A "pvxs::TypeCode::Int32A"
        Int64A "pvxs::TypeCode::Int64A"
        UInt8A "pvxs::TypeCode::UInt8A"
        UInt16A "pvxs::TypeCode::UInt16A"
        UInt32A "pvxs::TypeCode::UInt32A"
        UInt64A "pvxs::TypeCode::UInt64A"
        Float32A "pvxs::TypeCode::Float32A"
        Float64A "pvxs::TypeCode::Float64A"
        StringA "pvxs::TypeCode::StringA"
        StructA "pvxs::TypeCode::StructA"
        UnionA "pvxs::TypeCode::UnionA"
        AnyA "pvxs::TypeCode::AnyA"

    cdef enum StoreType:
        SNull "pvxs::StoreType::Null"
        SBool "pvxs::StoreType::Bool"
        SUInteger "pvxs::StoreType::UInteger"
        SInteger "pvxs::StoreType::Integer"
        SReal "pvxs::StoreType::Real"
        SString "pvxs::StoreType::String"
        SCompound "pvxs::StoreType::Compound"
        SArray "pvxs::StoreType::Array"

    cdef enum Kind:
        KBool     "Kind::Bool"
        KInteger  "Kind::Integer"
        Real      "Kind::Real"
        KString   "Kind::String"
        KCompound "Kind::Compound"
        KNull     "Kind::Null"

    cdef cppclass TypeCode:
        code_t code

        TypeCode()
        TypeCode(unsigned)
        TypeCode(code_t)

        bool valid() const
        #Kind kind() const
        unsigned order() const
        unsigned size() const
        bool isunsigned() const
        bool isarray() const

        StoreType storeAs() const

        TypeCode arrayOf() const
        TypeCode scalarOf() const

        const char* name() const

    cdef cppclass Member:
        Member()
        Member(TypeCode, const string&) except+
        #Member[IT](TypeCode, const string&, const IT&) except+
        #Member[IT](TypeCode, const string&, const string&, const IT&) except+

        void addChild(const Member& mem) except+


    cdef cppclass TypeDef:
        TypeDef()
        TypeDef(const TypeDef&)
        TypeDef(TypeDef&&)

        TypeDef(const Value&) except+

        TypeDef(TypeCode)
        # Cython has a problem with parameterized constructors
        #TypeDef[IT](TypeCode, const string&, const IT&) except+

        Member as(const string&) except+

        Value create() except+

    cdef cppclass Value:
        Value()
        Value(const Value&)
        Value(Value&&)

        Value cloneEmpty() except+
        Value clone() except+

        Value& assign(const Value&) except+

        Value allocMember() except+

        bool valid() const
        bool operator bool() const

        bool isMarked(bool, bool) const
        Value ifMarked(bool, bool) const

        void mark(bool) except+
        void unmark(bool, bool) except+

        TypeCode type() const
        StoreType storageType() const
        const string& id() except+
        bool idStartsWith(const string&) except+

        bool equalInst(const Value&)

        const string& nameOf(const Value&) except+

        void copyOut(void *ptr, StoreType type) except+
        bool tryCopyOut(void *ptr, StoreType type) except+
        void copyIn(const void *ptr, StoreType type) except+
        bool tryCopyIn(const void *ptr, StoreType type) except+

        V as[V]() except+
        bool as[V](V&) except+

        bool tryFrom[V](const V&) except+
        void _from "from" [V](const V&) except+

        Value& update[K,V](K, const V&) except+

        Value operator[](const string&) except+
        Value lookup(const string&) except+ KeyError

        size_t nmembers() const

        cppclass IAll:
            cppclass iterator:
                Value& operator*()
                iterator operator++()
                iterator operator--()
                iterator operator+(size_type)
                iterator operator-(size_type)
                #difference_type operator-(iterator)
                bint operator==(iterator)
                bint operator!=(iterator)
                bint operator<(iterator)
                bint operator>(iterator)
                bint operator<=(iterator)
                bint operator>=(iterator)
            iterator begin()
            iterator end()

        cppclass IChildren:
            cppclass iterator:
                Value& operator*()
                iterator operator++()
                iterator operator--()
                iterator operator+(size_type)
                iterator operator-(size_type)
                #difference_type operator-(iterator)
                bint operator==(iterator)
                bint operator!=(iterator)
                bint operator<(iterator)
                bint operator>(iterator)
                bint operator<=(iterator)
                bint operator>=(iterator)
            iterator begin()
            iterator end()

        cppclass IMarked:
            cppclass iterator:
                Value& operator*()
                iterator operator++()
                iterator operator--()
                iterator operator+(size_type)
                iterator operator-(size_type)
                #difference_type operator-(iterator)
                bint operator==(iterator)
                bint operator!=(iterator)
                bint operator<(iterator)
                bint operator>(iterator)
                bint operator<=(iterator)
                bint operator>=(iterator)
            iterator begin()
            iterator end()

        IAll iall() except+
        IChildren ichildren() except+
        IMarked imarked() except+
