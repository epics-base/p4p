Client Blocking API
===================

.. currentmodule:: p4p.client.thread

This module provides :py:class:`Context` for use in interactive and/or multi-threaded environment.
Most methods will block the calling thread until a return is available, or an error occurs.

Usage
-----

Start by creating a client :py:class:`Context`. ::

   >>> from p4p.client.thread import Context
   >>> Context.providers()
   ['pva', ....]
   >>> ctxt = Context('pva')

Get/Put
^^^^^^^

Get and Put operations can be performed on a single PV or a list of PVs. 
Get returns an NTtype which holds the value and common characteristics: 
severity, timestamp (seconds after epoch with nanoseconds), raw_stamp 
(seconds and nanoseconds) and status.::

   >>> V = ctxt.get('pv:name')
   >>> A, B = ctxt.get(['pv:1', 'pv:2'])
   >>> A.severity
   0
   >>> A.value
   0.0
   >>> ctxt.put('pv:name', 5)
   >>> ctxt.put('pv:name', {'value': 5}) # equivalent to previous
   >>> ctxt.put(['pv:1', 'pv:2'], [1, 2])

RPC
^^^

The RPC operation is similar to get+put except that the argument value must already
be a :py:class:`Value`. ::

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
        'path': 'pv:add',
        'query': {
            'lhs': 1,
            'rhs': 2,
        },
    })
    result = ctxt.rpc(V)

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
