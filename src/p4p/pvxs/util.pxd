#cython: language_level=2

from libc.stdint cimport uint8_t
from libcpp.string cimport string
from libcpp.map cimport map

cdef extern from "<pvxs/util.h>" namespace "pvxs" nogil:
    map[string, size_t] instanceSnapshot() except+

    cdef cppclass ServerGUID: # really derivies from std::array
        size_t size() const
        uint8_t& operator[](size_t)
        uint8_t* data()
