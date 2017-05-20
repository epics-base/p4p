Common Types
============

.. currentmodule:: p4p.nt

Helpers for creating standardized :py:class:`Type` instances.
as defined by http://epics-pvdata.sourceforge.net/alpha/normativeTypes/normativeTypes.html
.

.. _unwrap:

Automatic Value unwrapping
--------------------------

For convenience, various operations including :py:meth:`p4p.client.thread.Context.get`
can automatically transform the returned :py:class:`Value`.

For example, by default a NTScalar with a floating point value becomes :py:class:`scalar.ntfloat`
which behaves list the py:class:`float` type with some additional attributes

* timestamp - The update timestamp is a float representing seconds since 1 jan 1970 UTC.
* raw_stamp - A tuple of (seconds, nanosecons)
* severity - An integer in the range [0, 3]
* raw - The complete underlying :class:`~p4p.Value`

Controlling Unwrapping
^^^^^^^^^^^^^^^^^^^^^^

The *unwrap* argument of client Context controls whether unwrapping is done.
Possible values are None (the default), False (no unwrapping), or a dict with
additional/custom unwrappers.

This dictionary is keyed using the structure ID.
The value must be a callable which takes one argument,
which is a :class:`~p4p.Value`.

For example.
A simplified version of the default NTScalar unwrapping (which preserves meta-data)
would be. ::

   >>> C=Context('pva', unwrap={"epics:nt/NTScalar:1.0":lambda V:V.value})
   >>> C.get('pv:counter')
   5

Which extracts the 'value' field and discards all others.

To unwrap NTTable as an iterator of :class:`OrderedDict`. ::

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
