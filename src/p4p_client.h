#ifndef P4P_CLIENT_H
#define P4P_CLIENT_H

#include "p4p.h"

#include <pv/pvAccess.h>

struct Channel {
    POINTER_DEFINITIONS(Channel);

    ~Channel();

    struct Req : public epics::pvAccess::ChannelRequester {
        POINTER_DEFINITIONS(Req);

        Channel::weak_pointer owner;
        Req(const Channel::shared_pointer& o) : owner(o) {}
        virtual ~Req() {}

        virtual std::string getRequesterName() { return "p4p.Channel"; }

        virtual void channelCreated(const epics::pvData::Status& status, epics::pvAccess::Channel::shared_pointer const & channel);
        virtual void channelStateChange(epics::pvAccess::Channel::shared_pointer const & channel, epics::pvAccess::Channel::ConnectionState connectionState);
    };

    epics::pvAccess::Channel::shared_pointer channel;
};

struct MonitorOp {
    POINTER_DEFINITIONS(MonitorOp);
    typedef epics::pvAccess::Monitor operation_t;
    typedef epics::pvAccess::MonitorRequester requester_t;

    struct Req : public requester_t {
        POINTER_DEFINITIONS(Req);

        MonitorOp::weak_pointer owner;
        Req(const MonitorOp::shared_pointer& o) : owner(o) {}
        virtual ~Req() {}

        virtual std::string getRequesterName() { return "p4p.MonitorOp"; }

        virtual void monitorConnect(epics::pvData::Status const & status,
                                    epics::pvAccess::Monitor::shared_pointer const & monitor,
                                    epics::pvData::StructureConstPtr const & structure);

        virtual void monitorEvent(epics::pvAccess::Monitor::shared_pointer const & monitor);

        virtual void unlisten(epics::pvAccess::Monitor::shared_pointer const & monitor);

        virtual void channelDisconnect(bool destroy);
    };

    MonitorOp();
    ~MonitorOp();

    operation_t::shared_pointer op;

    /** error/non-empty callback
     * Called with one of:
     *   True: New data, call poll()
     *   False: no more new data ever (normal completion of stream)
     *   None: underlying channel disconnected.  Will reconnect
     *   Exception: some specific error
     */
    PyRef event;
    bool empty, done;

    void call_cb(PyObject *obj);
};

// common base for get/put/rpc
struct OpBase {
    POINTER_DEFINITIONS(MonitorOp);

    virtual ~OpBase();

    // called with GIL
    virtual void cancel();

    // completion callback
    PyRef cb;
    // put value
    PyRef pyvalue;
    // rpc value
    epics::pvData::PVStructure::shared_pointer pvvalue;

    void call_cb(PyObject *obj);

protected:
    OpBase() {}
};

struct GetOp : public OpBase {
    POINTER_DEFINITIONS(GetOp);
    typedef epics::pvAccess::ChannelGet operation_t;
    typedef operation_t::requester_type requester_t;

    struct Req : public requester_t {
        POINTER_DEFINITIONS(Req);

        GetOp::weak_pointer owner;
        Req(const GetOp::shared_pointer& o) : owner(o) {}
        virtual ~Req() {}

        virtual std::string getRequesterName() { return "p4p.GetOp"; }

        virtual void channelGetConnect(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelGet::shared_pointer const & channelGet,
            epics::pvData::Structure::const_shared_pointer const & structure);

        virtual void getDone(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelGet::shared_pointer const & channelGet,
            epics::pvData::PVStructure::shared_pointer const & pvStructure,
            epics::pvData::BitSet::shared_pointer const & bitSet);

        virtual void channelDisconnect(bool destroy);
    };

    operation_t::shared_pointer op;

    GetOp() {}
    virtual ~GetOp() {}

    virtual void cancel();
};

struct PutOp : public OpBase {
    POINTER_DEFINITIONS(PutOp);
    typedef epics::pvAccess::ChannelPut operation_t;
    typedef operation_t::requester_type requester_t;

    struct Req : public requester_t {
        POINTER_DEFINITIONS(Req);

        PutOp::weak_pointer owner;
        Req(const PutOp::shared_pointer& o) : owner(o) {}
        virtual ~Req() {}

        virtual std::string getRequesterName() { return "p4p.PutOp"; }

        virtual void channelPutConnect(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelPut::shared_pointer const & channelPut,
            epics::pvData::Structure::const_shared_pointer const & structure);

        virtual void putDone(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelPut::shared_pointer const & channelPut);

        virtual void getDone(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelPut::shared_pointer const & channelPut,
            epics::pvData::PVStructure::shared_pointer const & pvStructure,
            epics::pvData::BitSet::shared_pointer const & bitSet)
        { /* no used */ }

        virtual void channelDisconnect(bool destroy);
    };

    operation_t::shared_pointer op;

    PutOp() {}
    virtual ~PutOp() {}

    virtual void cancel();
};

struct RPCOp : public OpBase {
    POINTER_DEFINITIONS(RPCOp);
    typedef epics::pvAccess::ChannelRPC operation_t;
    typedef operation_t::requester_type requester_t;

    struct Req : public requester_t {
        POINTER_DEFINITIONS(Req);

        RPCOp::weak_pointer owner;
        Req(const RPCOp::shared_pointer& o) : owner(o) {}
        virtual ~Req() {}

        virtual std::string getRequesterName() { return "p4p.PutOp"; }

        virtual void channelRPCConnect(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelRPC::shared_pointer const & channelRPC);

        virtual void requestDone(
            const epics::pvData::Status& status,
            epics::pvAccess::ChannelRPC::shared_pointer const & channelRPC,
            epics::pvData::PVStructure::shared_pointer const & pvResponse);

        virtual void channelDisconnect(bool destroy);
    };

    operation_t::shared_pointer op;
    // sent once the network op may have gone out.
    // the point at which we can't retry safely
    bool sent;

    RPCOp() {}
    virtual ~RPCOp() {}

    virtual void cancel();
};


/* we wrap a shared_ptr<> here, but this is the only presisitent strong reference.
 * *::Req::owner is a weak ref. to the same
 */
typedef PyClassWrapper<std::tr1::shared_ptr<Channel> > PyChannel;
typedef PyClassWrapper<std::tr1::shared_ptr<MonitorOp> > PyMonitorOp;
typedef PyClassWrapper<std::tr1::shared_ptr<OpBase> > PyOp;


#endif // P4P_CLIENT_H
