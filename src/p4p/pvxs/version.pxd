#cython: language_level=2

from libcpp cimport bool

cdef extern from "<pvxs/version.h>" namespace "pvxs" nogil:
    enum: PVXS_VERSION
    enum: EPICS_VERSION_INT

    const char *version_str()
    unsigned long version_int()
    unsigned long version_abi_int()
    bool version_abi_check()
