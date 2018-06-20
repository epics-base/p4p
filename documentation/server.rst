Server API
==========

.. currentmodule:: p4p.server

Running a PVA Server
--------------------

The :py:class:`Server` starts/stops a PVAccess server.
However, by itself a server is not useful.
One or more Providers must be named to give the server
a useful function.

Two Provider containers are available: :py:class:`StaticProvider` or :py:class:`DynamicProvider`.
Both are used with one of the SharedPV classes: :py:class:`raw.SharedPV`,
:py:class:`thread.SharedPV`, :py:class:`asyncio.SharedPV`, and/or :py:class:`cothread.SharedPV`.
These different threading models may be mixed into a single Provider.

.. note:: It is recommended to prefer :py:class:`StaticProvider` and leave :py:class:`DynamicProvider` for special cases.

DynamicProvider Handler Interface
---------------------------------

A :py:class:`DynamicProvider` Handler class will define the following:

.. class:: ProviderHandler

    .. method:: testChannel(pvname)

        Called with a PV name which some client is searching for.

        :return: True to claim this PV.

    .. method:: makeChannel(pvname, src):

        Called when a client attempts to create a Channel for some PV.
        The object which is returned will not be collected until
        the client closes the Channel or becomes disconnected.

        :return: A :py:class:`SharedPV` instance.


SharedPV Handler Interface
--------------------------

A :py:class:`SharedPV` Handler class will

.. class:: SharedPVHandler

    .. method:: rpc(pv, op)

        Called each time a client issues a Remote Procedure Call
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param ServerOperation op: The operation being initiated.

    .. method:: put(pv, op)

        Called each time a client issues a Put
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param ServerOperation op: The operation being initiated.

    .. method:: onFirstConnect(pv)

        Called when the first Client channel is created.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.

    .. method:: onLastDisconnect(pv)

        Called when the last Client channel is closed.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.


API Reference
-------------

.. autoclass:: Server

    .. automethod:: conf

    .. automethod:: stop

.. autoclass:: StaticProvider

    .. automethod:: close

    .. automethod:: add

    .. automethod:: remove

.. autoclass:: DynamicProvider

.. autofunction:: installProvider

.. autofunction:: removeProvider

.. autoclass:: ServerOperation

    .. automethod:: pvRequest

    .. automethod:: value

    .. automethod:: done

    .. automethod:: info

    .. automethod:: warn


.. currentmodule:: p4p.server.raw

.. autoclass:: SharedPV

    .. automethod:: open

    .. automethod:: close

    .. automethod:: post


There is a SharedPV class for each of the four threading models.
All have the same methods as :py:class:`raw.SharedPV`.

.. currentmodule:: p4p.server.thread

.. autoclass:: SharedPV


.. currentmodule:: p4p.server.asyncio

.. autoclass:: SharedPV


.. currentmodule:: p4p.server.cothread

.. autoclass:: SharedPV
