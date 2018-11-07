
.. _clientapi:

Client API
==========

.. currentmodule:: p4p.client.thread

This module provides :py:class:`Context` for use in interactive and/or multi-threaded environment.
Most methods will block the calling thread until a return is available, or an error occurs.

Two alternatives to `p4p.client.thread.Context` are provided
`p4p.client.cothread.Context` and `p4p.client.asyncio.Context`.
These differ in how blocking for I/O operation is performed,
and the environment in which Monitor callbacks are run.

Usage
-----

Start by creating a client :py:class:`Context`. ::

   >>> from p4p.client.thread import Context
   >>> Context.providers()
   ['pva', ....]
   >>> ctxt = Context('pva')

.. note:: The default network configuration taken from the process environment
          may be overridden by passing 'conf=' to the `Context` class constructor.

See `overviewpva` for background on PVAccess protocol.

Get/Put
^^^^^^^

Get and Put operations can be performed on single PVs or a list of PVs. ::

   >>> V = ctxt.get('pv:name')
   >>> A, B = ctxt.get(['pv:1', 'pv:2'])
   >>> ctxt.put('pv:name', 5)
   >>> ctxt.put('pv:name', {'value': 5}) # equivalent to previous

By default the values returned by :py:meth:`Context.get` are subject to :py:ref:`unwrap`.

Monitor
^^^^^^^

Unlike get/put/rpc, the :py:meth:`Context.monitor` method does not block.
Instead it accepts a callback function which is called with each
new :py:class:`Value`, or :py:class:`Exception`. ::

   def cb(V):
          print 'New value', V
   sub = ctxt.monitor('pv:name', cb)
   time.sleep(10) # arbitrary wait
   sub.close()

The monitor method returns a :py:class:`Subscription` which has a close method
to end the subscription.

By default the values passed to monitor callbacks are subject to :py:ref:`unwrap`.

`p4p.client.thread.Context` Runs callbacks in a worker thread pool.

`p4p.client.cothread.Context` Runs callbacks in a per-subscription cothread.

`p4p.client.asyncio.Context` Runs callbacks in a per-subscription coroutine.

In all cases it is safe for a callback to block/yield.
Subsequent updates for a `Subscription` will not be delivered until the current callback has completed.
However, updates for other Subscriptions may be delivered.

RPC
^^^

See `rpcapi`.

API Reference
-------------

.. module:: p4p.client.thread

.. autoclass:: Context

    .. autoattribute:: name

    .. automethod:: close

    .. automethod:: get

    .. automethod:: put

    .. automethod:: monitor

    .. automethod:: rpc

    .. automethod:: providers

    .. automethod:: disconnect

    .. automethod:: set_debug

.. autoclass:: Subscription

    .. automethod:: close

.. autoclass:: Disconnected

.. autoclass:: RemoteError

.. autoclass:: Cancelled

.. autoclass:: Finished

.. autoclass:: TimeoutError
