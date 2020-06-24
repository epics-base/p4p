#cython: language_level=2

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

cdef extern from "<pvxs/sharedArray.h>" namespace "pvxs" nogil:
    cdef enum ArrayType:
        Null "ArrayType::Null"
        Bool "pvxs::ArrayType::Bool"
        Int8 "pvxs::ArrayType::Int8"
        Int16 "pvxs::ArrayType::Int16"
        Int32 "pvxs::ArrayType::Int32"
        Int64 "pvxs::ArrayType::Int64"
        UInt8 "pvxs::ArrayType::UInt8"
        UInt16 "pvxs::ArrayType::UInt16"
        UInt32 "pvxs::ArrayType::UInt32"
        UInt64 "pvxs::ArrayType::UInt64"
        Float32 "pvxs::ArrayType::Float32"
        Float64 "pvxs::ArrayType::Float64"
        String "pvxs::ArrayType::String"
        Value "pvxs::ArrayType::Value"

    size_t elementSize(ArrayType type) except+

    cdef cppclass shared_array[T]:
        shared_array()
        shared_array(const shared_array&)
        shared_array(shared_array&&)

        shared_array(size_t cnt) except+
        shared_array(T*, size_t)
        shared_array(const shared_ptr[T]&, size_t)
        # TODO omitted

        size_t size() const
        bool empty() const
        bool unique() const
        size_t clear()
        void swap(const shared_array&)

        T* data() const
        const shared_ptr[T]& dataPtr() const

        void resize(size_t cnt) except+
        void make_unique() except+

        shared_array[C] freeze[C]() except+
        shared_array[C] castTo[C]() except+
        shared_array[C] convertTo[C]() except+

        T* begin()
        T* end()

        T& operator[](size_t)
        T& at(size_t) except+

        # void only
        ArrayType original_type() const
