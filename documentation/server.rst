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
These can be used with :py:class:`SharedPV`.

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

Example RPC Provider
--------------------

This example provided for information purposes.
See the :py:mod:`rpc` module for more functional RPC handling and dispatch.

>>> from p4p.nt import NTScalar
>>> class ExampleProvider(object):
        def __init__(self, myname):
            self.name = name
            # prepare return type
            self.addret = NTScalar.buildType('d')
        def testChannel(self, name):
            return name==self.name
        def makeChannel(self, name, src):
            # we need no per-channel state, so re-use the provider as the channel
            return self is name==self.name else None
        def rpc(self, response, request):
            # Real scalable provider will do this from a worker thread
            V = Value(self.addret, {
                'value': float(request.query.lhs) + float(request.query.rhs),
            })
            response.done(reply=V)

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

