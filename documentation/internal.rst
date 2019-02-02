Internal API
============

This section is intended to help P4P developers.
The API described is *not* considered a public or stable API.
It may change without notice.

Ownership
---------

The ownership relations between the various objects in the C++
extensions are shown in the following diagrams.

* Python objects are blue ovals
* C++ objects are black boxes

* red lines are shared_ptr<>
* green lines are weak_ptr<>
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

API Reference
-------------

Raw client API
~~~~~~~~~~~~~~

.. currentmodule:: p4p.client.raw

.. automodule:: p4p.client.raw
    :members:

Raw server API
~~~~~~~~~~~~~~

.. currentmodule:: p4p.server.raw

.. automodule:: p4p.server.raw
    :members:

C Extension
~~~~~~~~~~~

.. currentmodule:: p4p._p4p

.. automodule:: p4p._p4p
    :members:
    :undoc-members:

    .. autoclass:: Array

    .. autofunction:: clearProviders

