.. _ntapi:

Common Types
============

.. currentmodule:: p4p.nt

Helpers for creating standardized :py:class:`Type` instances.
as defined by http://epics-pvdata.sourceforge.net/alpha/normativeTypes/normativeTypes.html
.

.. _unwrap:

Automatic Value unwrapping
--------------------------

Automatic transformation can be performed. between `Value` and more convenient types.

Transformation may be performed at the following points:

* The result of `p4p.client.thread.Context.get()`,
* The argument the callable passed to `p4p.client.thread.Context.monitor()`
* The argument of `p4p.client.thread.Context.put()`
* The argument of `p4p.client.thread.Context.rpc()`
* The argument of `p4p.server.thread.SharedPV.open()`
* The argument of `p4p.server.thread.SharedPV.post()`
* The result of `p4p.server.thread.SharedPV.current()`

Controlling (Un)wrapping
^^^^^^^^^^^^^^^^^^^^^^^^

Client `p4p.client.thread.Context` accepts an argument nt= which may be
`None` to sure some reasonable defaults.  `False` disables wrapping,
and always works with `Value`.  *nt=* may also be passed a dictionary
keyed by top level structure IDs mapped to callables returning objects
conforming to `WrapperInterface`.

The *unwrap* argument is legacy which functions like *nt=* but
mapping to plain functions instead of wrapper objects. ::

    from p4p.client.thread import Context
    ctxt=Context('pva', nt=False) # disable (un)wrap.  All methods use Value

Server `p4p.server.thread.SharedPV` accepts an argument *nt=* which
is an instance of an object conforming to `WrapperInterface`. ::

    from p4p.server.thread import SharedPV
    from p4p.nt import NTScalar
    pv1 = SharedPV() # pv1.open() expects a Value
    pv2 = SharedPV(nt=NTScalar('d'))
    pv2.open(4.2) # NTScalar automatically wraps this float into a Value

Conforming objects include `NTScalar`, `NTNDArray`, and others listed below.

.. autofunction:: defaultNT

NT wrap/unwrap interface
^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: WrapperInterface

    :since: 3.1.0

    .. classmethod:: buildtype()

        Returns a `Type` based on some helper specific conditions.

        :rtype: `Type`

    .. method:: __init__

        Each time the type ID of a Channel changes, a new wrapper will be instantiated if available.

    .. method:: unwrap(Value) -> object

        Called with a `Value` and may return an arbitrary object.

        Called by both clients and servers.  eg. during `p4p.client.thread.Context.get()`
        and `p4p.server.thread.SharedPV.current()`.

    .. method:: wrap(object) -> Value

        Called with an arbitrary object which it should try to translate into a `Value`.

        Called by servers.  eg. during `p4p.server.thread.SharedPV.post()`.

    .. method:: assign(Value, object)

        Called to update a `Value` based on an arbitrary object.

        Called by clients.  eg. during `p4p.client.thread.Context.put()`, where
        the get= argument effects the state of the `Value` passed in.

API Reference
-------------

.. autoclass:: NTScalar

    .. automethod:: buildType

    .. automethod:: wrap

    .. automethod:: assign

    .. automethod:: unwrap

.. autoclass:: NTNDArray

    .. automethod:: buildType

    .. automethod:: wrap

    .. automethod:: assign

    .. automethod:: unwrap

.. autoclass:: NTTable

    .. automethod:: buildType

    .. automethod:: wrap

    .. automethod:: unwrap

.. autoclass:: NTURI

    .. automethod:: buildType

    .. automethod:: wrap

.. autoclass:: NTMultiChannel

    .. automethod:: buildType

.. currentmodule:: p4p.nt.scalar

.. autoclass:: ntfloat

.. autoclass:: ntint

.. autoclass:: ntstr

.. autoclass:: ntnumericarray

.. autoclass:: ntstringarray

.. currentmodule:: p4p.nt.ndarray

.. autoclass:: ntndarray
