.. _rpcapi:

RPC Server Helpers
==================

.. currentmodule:: p4p.rpc

Server Example
--------------

Remote Procedure Calls are received by the methods of a "target" object.
This is any class which has method decorated with :py:func:`rpc`.

For example: ::

    from p4p.rpc import rpc, quickRPCServer
    from p4p.nt import NTScalar
    class Summer(object):
        @rpc(NTScalar("d"))
        def add(self, lhs, rhs): # 'lhs' and 'rhs' are arbitrary names.  The method name 'add' will be part of the PV name
            return float(lhs) + float(rhs)
    adder = Summer()

Turn on logging to see RPC related errors. ::

    import logging
    logging.basicConfig(level=logging.DEBUG)

Now run a server with :func:`quickRPCServer`. ::

    quickRPCServer(provider="Example", 
                   prefix="pv:call:",  # A prefix for method PV names.
                   target=adder)

At this point the server is active.

Beginning with EPICS 7.0.2 this can be tested using the "pvcall" utility

.. code-block:: sh

    $ pvcall pv:call:add lhs=1 rhs=1
    2018-10-30 19:49:34.834  2

Previous to EPICS 7.0.2 this can be tested using the "eget" utility from the pvAccessCPP module.

.. code-block:: sh

    $ eget -s pv:call:add -a lhs=1 -a rhs=1
    2
    $ eget -s pv:call:add -a lhs=1 -a rhs=1 -N  # for more detail
    epics:nt/NTScalar:1.0 
        double value 2
        alarm_t alarm NO_ALARM NO_STATUS <no message>
        time_t timeStamp 2017-05-20T08:14:31.917 0

Client Example
--------------

Remote procedure calls are made through the :meth:`~p4p.client.thread.Context.rpc` method of a :class:`~p4p.client.thread.Context`.
To assist in encoding arguments, a proxy object can be created with the :func:`rpcproxy` decorator.
A proxy for the preceding example would be: ::

    from p4p.rpc import rpcproxy, rpccall
    @rpcproxy
    class MyProxy(object):
        @rpccall('%sadd')
        def add(lhs='d', rhs='d'):
            pass

This proxy must be associated with a Context. ::

    from p4p.client.thread import Context
    ctxt = Context('pva')
    proxy = MyProxy(context=ctxt, format='pv:call:')
    print proxy.add(1, 1)


A decorated proxy class has two additional contructor arguments.

Using Low Level Client API
~~~~~~~~~~~~~~~~~~~~~~~~~~

It may be helpful to illustrate what a proxy method call is actually doing. ::

    from p4p import Type, Value
    V = Value(Type([
        ('schema', 's'),
        ('path', 's'),
        ('query', ('s', None, [
            ('lhs', 'd'),
            ('rhs', 'd'),
        ])),
    ]), {
        'schema': 'pva',
        'path': 'pv:call:add',
        'query': {
            'lhs': 1,
            'rhs': 1,
        },
    })
    print ctxt.rpc('pv:call:add', V)

API Reference
-------------

.. autofunction:: rpc

.. autofunction:: rpcproxy

.. autofunction:: rpccall

.. autofunction:: quickRPCServer

.. autoclass:: RPCProxyBase

  :param Context context: The client :class:`~p4p.client.thread.Context` through which calls are made
  :param format: A tuple or dict which is applied with the format '%' operator to the name strings given to :func:`rpccall`.

  .. autoattribute:: context

  .. autoattribute:: timeout

  .. autoattribute:: authority

  .. autoattribute:: throw

.. autoclass:: NTURIDispatcher

.. autoclass:: RemoteError

.. autoclass:: WorkQueue

  .. automethod:: handle

  .. automethod:: interrupt
