Overview
========

.. currentmodule:: p4p

What is EPICS?
~~~~~~~~~~~~~~

See https://epics.anl.gov/

.. _overviewpva:

What is PVAccess?
~~~~~~~~~~~~~~~~~

The PVAccess network protocol is a hybrid supporting request/response,
and publish/subscribe operations.

PVA is closely related to the Channel Access (CA) protocol,
which may work along side, and is intended to supersede.

Four protocol operations are supported through the P4P wrapper.

- Get - Fetch the present value of a PV.
- Put - Change the value of a PV.
- Monitor - Subscribe to changes in the value of a PV.
- RPC - A remote method call.

Get, Put, Monitor, and RPC are to the PVA protocol what GET, PUT, POST are to the HTTP protocol.

What is a PV?
~~~~~~~~~~~~~

In the EPICS world a Process Variable (PV) refers to the idea of
a globally addressed data structure.  An EPICS control system is
composed of many PVs (in the millions for large facilities).  The present value of
a PV is modified by a combination of remote operations via CA
and/or PVA, and via local processing (eg. values read from local
hardware).

A common example of a PV is a measurement value, for example
a temperature measured by a particular sensor.

Another example would be an electro-mechanical relay, which may be opened or closed.

In this case a Get operation would poll the current open/closed state of the relay.
A Monitor operation (subscription) would receive notification when the relay state changes.
A Put operation would be used to command the relay to open or close, or perhaps toggle (the precise meaning of a Put is context dependent).

So the Get, Put, and Monitor operation on a given PV are conventionally operating on a common data structure.
The RPC operation is more arbitrary, and need not have any relationship with a common data structure (eg. the open/closed state of the relay.)

.. note:: In the context of the PVA or CA protocols, a **"PV name"** is an address string which uniquely identifies a Process Variable.
          All PVA network operations begin with a "PV name" string.

A "PV name" string is to the PVA and CA protocols what a URL is to the HTTP protocol.
The main difference being that while a URL is hierarchical, having a hostname and path string,
a PV name is not.  The namespace of PV names is by default all local IP subnets (broadcast domains).
This can be made more complicated though the specifics of client/server network configuration.

The P4P module provides the ability to run PVA clients (:ref:`clientapi`) and/or servers (:ref:`serverapi`).
Additional convenience APIs are provided when using the RPC network operation (:ref:`rpcapi`). 

What is a Value?
~~~~~~~~~~~~~~~~

P4P represents the data which goes into, and results from, PVA network operations
with the :py:class:`Value` class which represents a strongly typed data structure.
See :ref:`valueapi` for details.

A set of standardized data structure definitions, and utilities is provided as the :ref:`ntapi` API.
