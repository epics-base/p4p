Common Types
============

.. currentmodule:: p4p.nt

Helpers for creating standardized :py:class:`Type` instances.
as defined by http://epics-pvdata.sourceforge.net/alpha/normativeTypes/normativeTypes.html
.

These helpers have one or more of the following.

A static method *buildType()* which returns a :class:`~p4p.Type` based on some helper specific conditions.

An instance of a helper takes the same, or similar, arguments as the *buildType()*.
In fact, the result of *buildType()* is stored in an attribute named *type*.
The instance may have a method *wrap()* which takes some python types and stores them in the returned :class:`~p4p.Value`.
This may be used with the :class:`p4p.rpc.rpc` decorator to specify the return type. ::

    from p4p.rpc import rpc
    class MyRPCServer(object):
        @rpc(NTScalar("d"))
        def magicnumber(self):
            return 4

In this example, the returned *4* is passed to :method:`NTScalar.wrap` which stores it as a double precision float.

In addition, NT helpers may have one, or more, method unwrap*() which accept a :class:`~p4p.Value` and return
some python value.

.. _unwrap:

Automatic Value unwrapping
--------------------------

For convenience, various operations including :py:meth:`p4p.client.thread.Context.get`
can automatically transform the returned :py:class:`~p4p.Value`.

For example, by default a NTScalar with a floating point value becomes :py:class:`scalar.ntfloat`
which behaves list the :class:`float` type with some additional attributes

* .timestamp - The update timestamp is a float representing seconds since 1 jan 1970 UTC.
* .raw_stamp - A tuple of (seconds, nanoseconds)
* .severity - An integer in the range [0, 3]
* .raw - The complete underlying :class:`~p4p.Value`

Controlling Unwrapping
^^^^^^^^^^^^^^^^^^^^^^

The *unwrap* argument of client Context controls whether unwrapping is done.
Possible values are None (the default), False (no unwrapping), or a dict with
additional/custom unwrappers.

This dictionary is keyed using the structure ID.
The value must be a callable which takes one argument,
which is a :class:`~p4p.Value`.

For example.
A simplified version of the default NTScalar unwrapping which discards meta-data
would be. ::

   >>> C=Context('pva', unwrap={"epics:nt/NTScalar:1.0":lambda V:V.value})
   >>> C.get('pv:counter')
   5

Which extracts the 'value' field and discards all others.

To unwrap NTTable as an iterator yielding :class:`OrderedDict`. ::

   >>> C=Context('pva', unwrap={"epics:nt/NTTable:1.0":p4p.nt.NTTable.unwrap})
   >>> for row in C.rpc('pv:name', ....):
        print(row)

API Reference
-------------

.. autoclass:: NTScalar

    .. automethod:: buildType

    .. automethod:: wrap

    .. automethod:: unwrap

.. autoclass:: NTMultiChannel

    .. automethod:: buildType

.. autoclass:: NTTable

    .. automethod:: buildType

    .. automethod:: wrap

    .. automethod:: unwrap

.. autoclass:: NTURI

    .. automethod:: buildType

    .. automethod:: wrap

.. currentmodule:: p4p.nt.scalar

.. autoclass:: ntfloat

.. autoclass:: ntint

.. autoclass:: ntstr

.. autoclass:: ntnumericarray

.. autoclass:: ntstringarray
