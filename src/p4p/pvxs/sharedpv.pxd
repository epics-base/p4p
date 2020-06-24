#cython: language_level=2

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp.map cimport map

from .data cimport Value
from .source cimport Source

cdef extern from "<pvxs/sharedpv.h>" namespace "pvxs::server" nogil:
    cdef cppclass SharedPV:
        @staticmethod
        SharedPV buildMailbox() except+
        @staticmethod
        SharedPV buildReadonly() except+

        SharedPV()

        #void attach(std::unique_ptr<ChannelControl>&& op);
        #void onFirstConnect(std::function<void()>&& fn);
        #void onLastDisconnect(std::function<void()>&& fn);
        #void onPut(std::function<void(SharedPV&, std::unique_ptr<ExecOp>&&, Value&&)>&& fn);
        #void onRPC(std::function<void(SharedPV&, std::unique_ptr<ExecOp>&&, Value&&)>&& fn);

        void open(const Value& initial) except+
        bool isOpen() except+
        void close() except+

        void post(const Value& val) except+
        void fetch(Value& val) except+
        Value fetch() except+

    cdef cppclass StaticSource:
        @staticmethod
        StaticSource build() except+;

        StaticSource()

        shared_ptr[Source] source() except+

        void close() except+
        StaticSource& add(const string& name, const SharedPV& pv) except+
        StaticSource& remove(const string& name) except+

        map[string, SharedPV] list() except+
