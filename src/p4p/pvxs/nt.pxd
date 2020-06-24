#cython: language_level=2

from libcpp cimport bool

from . cimport data

cdef extern from "<pvxs/nt.h>" namespace "pvxs::nt" nogil:
    cdef cppclass NTScalar:
        data.TypeCode value
        bool display
        bool control
        bool valueAlarm

        data.TypeDef build() except+
        data.Value create() except+
