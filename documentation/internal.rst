Internal API
============

This section is intended to help P4P devlopers.
The API described is _not_ considered a public or stable API.
It may change without notice.

Ownership
---------

The ownership relations between the various objects in the C++
extensions are shown in the following diagrams.

* Python objects have blue blue oval
* C++ objects are black boxes

* red lines are shared_ptr<>
* grean lines are weak_ptr<>
* blue lines are python refs. (aka stored PyObject*)
* dashed lines are explicit breaks of a ref. loop

Type and Value
~~~~~~~~~~~~~~

.. digraph:: values

        Value [shape=oval, color=blue];
        Type [shape=oval, color=blue];
        Array [shape=oval, color=blue];
        ndarray [shape=oval, color=blue];
        PVStructure [shape=box];
        Structure [shape=box];
        shared_vector [shape=box];

        # wraps
        Value -> PVStructure [color=red];
        Type -> Structure [color=red];
        Array -> shared_vector [color=red];
        # internal
        PVStructure -> shared_vector [color=red];
        PVStructure -> Structure [color=red];
        # pyrefs
        ndarray -> Array [color=blue];

Client
~~~~~~

.. digraph:: client

        Context [shape=oval, color=blue];
        ChannelProvider [shape=box];
        Context -> ChannelProvider [color=red];

        PyChannel [shape=oval, color=blue, label="Channel"];
        Channel [shape=box];
        PyChannel -> Channel [color=red];

        ChannelReq [shape=box];
        Channel -> ChannelReq [color=red];
        ChannelReq -> PyChannel [color=green];

        Channel -> ChannelProvider [color=red];

        Operation [shape=oval, color=blue];

        OpCallback [shape=oval, color=blue,label="callback"];
        Operation -> OpCallback [color=blue];

        ChannelGet [shape=box,label="ChannelGet/Put/RPC"];
        Operation -> ChannelGet [color=red];
        ChannelGet -> Channel [color=red];

        GetReq [shape=box,label="ChannelGet/Put/RPCRequester"];
        GetReq -> Operation [color=green];
        ChannelGet -> GetReq [color=red];

        Monitor [shape=box];
        MonitorRequester [shape=box];
        Subscription [shape=oval, color=blue];
        handler [shape=oval, color=blue];
        Subscription -> Monitor [color=red];
        Monitor -> MonitorRequester [color=red];
        Monitor -> Channel [color=red];
        MonitorRequester -> Subscription [color=green];
        Subscription -> handler [color=blue];

Server
~~~~~~

.. digraph:: server

        PyServerProvider [shape=box];
        provider [shape=oval, color=blue];
        PyServerProvider -> provider [color=blue];

        PyServerChannel [shape=box];
        channel [shape=oval, color=blue];
        ChannelRequester [shape=box];
        PyServerChannel -> channel [color=blue];
        PyServerChannel -> ChannelRequester [color=red];
        PyServerChannel -> PyServerProvider [color=red];

        PyServerRPC [shape=box];
        ChannelRPCRequester [shape=box];
        PyServerRPC -> ChannelRPCRequester [color=red];
        PyServerRPC -> PyServerChannel [color=red];

        RPCReply [shape=oval, color=blue];
        channel [shape=oval, color=blue];
        RPCReply -> PyServerRPC [color=red];
        PyServerRPC -> RPCReply [color=green];

API Reference
-------------

.. currentmodule:: p4p._p4p

.. automodule:: p4p._p4p
    :members:
    :undoc-members:

    .. autoclass:: Array

    .. autoclass:: RPCReply

    .. autoclass:: Type

    .. autoclass:: Value
