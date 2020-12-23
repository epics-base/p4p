#cython: language_level=2

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.list cimport list as clist
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
#from libcpp.functional cimport function

from .data cimport Value
from .util cimport ServerGUID
from .source cimport ClientCredentials

cdef extern from "<pvxs/client.h>" namespace "pvxs::client" nogil:
    ctypedef map[string, string] Config_defs_t "pvxs::client::Config::defs_t"

    cdef cppclass Config:
        vector[string] addressList
        vector[string] interfaces
        unsigned short udp_port
        bool autoAddrList

        @staticmethod
        Config fromEnv() except+
        @staticmethod
        Config fromDefs(const map[string, string]&) except+

        Config& applyEnv() except+
        Config& applyDefs(const Config_defs_t&) except+
        void updateDefs(Config_defs_t& defs) except+

        void expand() except+

        Context build() except+

    enum cacheAction "pvxs::client::Context::cacheAction":
        Clean "pvxs::client::Context::Clean"
        Drop "pvxs::client::Context::Drop"
        Disconnect "pvxs::client::Context::Disconnect"

    cdef cppclass Context:
        Context()
        Context(const Config&) except+

        # methods returning builds must be except+ to workaround
        # https://github.com/cython/cython/issues/3759

        Config config() except+

        GetBuilder get(const string& pvname) except+
        PutBuilder put(const string& pvname) except+
        RPCBuilder rpc(const string& pvname) except+
        RPCBuilder rpc(const string& pvname, const Value&) except+
        MonitorBuilder monitor(const string& pvname) except+

        @staticmethod
        RequestBuilder request() except+

        void hurryUp() except+
        void cacheClear(const string& name, cacheAction action) except+
        void ignoreServerGUIDs(const vector[ServerGUID]& guids) except+

        Report report() except+

    cdef cppclass GetBuilder:
        GetBuilder& field(const string& fld) except+
        GetBuilder& record[T](const string& name, const T& val) except+
        GetBuilder& pvRequest(const string& fld) except+
        GetBuilder& rawRequest(const Value& fld) except+

        #GetBuilder& result(function[void(Result&&)]&& cb) except+

        shared_ptr[Operation] exec_ "exec"() except+

    cdef cppclass PutBuilder:
        PutBuilder& field(const string& fld) except+
        PutBuilder& record[T](const string& name, const T& val) except+
        PutBuilder& pvRequest(const string& fld) except+
        PutBuilder& rawRequest(const Value& fld) except+

        PutBuilder& fetchPresent(bool f)
        #PutBuilder& build(std::function[Value(Value&&)]&& cb)
        #PutBuilder& result(function[void(Result&&)]&& cb) except+

        shared_ptr[Operation] exec_ "exec"() except+

    cdef cppclass RPCBuilder:
        RPCBuilder& field(const string& fld) except+
        RPCBuilder& record[T](const string& name, const T& val) except+
        RPCBuilder& pvRequest(const string& fld) except+
        RPCBuilder& rawRequest(const Value& fld) except+

        #RPCBuilder& result(function[void(Result&&)]&& cb) except+

        shared_ptr[Operation] exec_ "exec"() except+

    cdef cppclass MonitorBuilder:
        MonitorBuilder& field(const string& fld) except+
        MonitorBuilder& record[T](const string& name, const T& val) except+
        MonitorBuilder& pvRequest(const string& fld) except+
        MonitorBuilder& rawRequest(const Value& fld) except+

        #MonitorBuilder& event(function[void(Subscription&)]&& cb)
        MonitorBuilder& maskConnected(bool m)
        MonitorBuilder& maskDisconnected(bool m)

        shared_ptr[Subscription] exec_ "exec"() except+

    cdef cppclass RequestBuilder:
        RequestBuilder& field(const string& fld) except+
        RequestBuilder& record[T](const string& name, const T& val) except+
        RequestBuilder& pvRequest(const string& fld) except+

        Value build() except+

    cdef cppclass Subscription:
        bool cancel() except+
        void pause(bool p) except+
        void resume() except+
        Value pop() except+

    cdef cppclass Result:
        pass

    cdef cppclass Operation:
        bool cancel() except+


    # really netcommon.h, but can't include this directly
    cdef cppclass ReportInfo "pvxs::client::ReportInfo":
        pass

    cdef cppclass Channel "pvxs::client::Report::Channel":
        string name
        size_t tx
        size_t rx
        shared_ptr[const ReportInfo] info

    cdef cppclass Connection "pvxs::client::Report::Connection":
        string peer
        shared_ptr[ClientCredentials] credentials
        size_t tx
        size_t rx
        clist[Channel] channels

    cdef cppclass Report:
        clist[Connection] connections
