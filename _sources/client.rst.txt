
.. _clientapi:

Client API
==========

.. currentmodule:: p4p.client.thread

This module provides :py:class:`Context` for use in interactive and/or multi-threaded environment.
Most methods will block the calling thread until a return is available, or an error occurs.

Alternatives to `p4p.client.thread.Context` are provided
`p4p.client.cothread.Context`,
`p4p.client.asyncio.Context`,
and `p4p.client.Qt.Context`.
These differ in how blocking for I/O operation is performed,
and the environment in which Monitor callbacks are run.

Note that `p4p.client.Qt.Context` behaves differently from the others in some respects.
This is described in `qtclient`_.

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
   >>> ctxt.put('pv:name', {'field_1.value': 5, 'field_2.value': 5}) # put to multiple fields

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

.. _qtclient:

Qt Client
---------

`p4p.client.Qt.Context` exists to bring the results of network operations into a Qt event loop.
This is done through the native signals and slots mechanism.

Use requires the optional dependency `qtpy <https://github.com/spyder-ide/qtpy>`_ package.

This dependency is expressed as an extras_require= of "qt".
It may be depended upon accordingly as "p4p[qt]".

`p4p.client.Qt.Context` differs from the other Context classes in several respects.

* Each Context attempts to minimize the number of subscriptions to each named PV.
  Multiple calls to monitor() will attempt to share this subscription if possible (subject to request argument).

* All monitor() calls must express a desired maximum update rate limit through the limitHz argument.

* As a convienence, the objects returned by put() and monitor() do not have to be stored by the caller.
  The internal references kept by the Context may be cleared through the disconnect() method.
  This cache extends to a single put and a single monitor subscription per PV.
  So eg. initiating a put() to a PV will implicitly cancel a previous in-progress put().
