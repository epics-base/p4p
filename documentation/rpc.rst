RPC Server Helpers
==================

.. currentmodule:: p4p.rpc

Basic Usage
-----------

Remote Procedure Calls are made through the methods of a "target" object.
This is any class which has method decorated with :py:func:`rpc`.

For example: ::

    from p4p.rpc import rpc, quickRPCServer
    from p4p.nt import NTScalar
    class Summer(object):
        @rpc(NTScalar("d"))
        def add(self, lhs, rhs):
            return float(lhs) + float(rhs)
    adder = Summer()

Turn on logging to see RPC related errors. ::

    import logging
    logging.basicConfig(level=logging.DEBUG)

Now run a server with :func:`quickRPCServer`. ::

    quickRPCServer(provider="Example", 
                   prefix="pv:call:",
                   target=adder)

At this point the server is active.
This can be tested using the "eget" utility from the pvAccessCPP module.

.. code-block:: sh

    $ eget -s pv:call:add -a lhs=1 -a rhs=1
    2
    $ eget -s pv:call:add -a lhs=1 -a rhs=1 -N
    epics:nt/NTScalar:1.0 
        double value 2
        alarm_t alarm NO_ALARM NO_STATUS <no message>
        time_t timeStamp 2017-05-20T08:14:31.917 0

API Reference
-------------

.. autofunction:: rpc

.. autofunction:: quickRPCServer

.. autoclass:: NTURIDispatcher

.. autoclass:: RemoteError

.. autoclass:: WorkQueue

  .. automethod:: handle

  .. automethod:: interrupt
