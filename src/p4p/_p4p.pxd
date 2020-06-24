#cython: language_level=2

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from pvxs cimport client
from pvxs cimport source
from pvxs cimport server

cdef class ClientProvider:
    cdef client.Context ctxt

cdef class Source:
    cdef string name
    cdef shared_ptr[source.Source] src


cdef class Server:
    cdef server.Server serv
    cdef object __weakref__
