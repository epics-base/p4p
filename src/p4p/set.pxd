# stripped down version of libcpp/set.pxd
# only supports const iterator (which is the only sane way to iterate a set)
cdef extern from "<set>" namespace "std" nogil:
    cdef cppclass set[T]:
        ctypedef T value_type
        cppclass const_iterator:
            T& operator*()
            const_iterator operator++()
            const_iterator operator--()
            bint operator==(const_iterator)
            bint operator!=(const_iterator)
        set() except +
        set(set&) except +
        const_iterator begin()
        void clear()
        size_t count(const T&)
        bint empty()
        const_iterator end()
        const_iterator find(T&)
        size_t size()
        void swap(set&)

