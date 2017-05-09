Server API
==========

.. currentmodule:: p4p.server

Running a PVA Server
--------------------

The :py:class:`Server` starts/stops a PVAccess server.
However, by itself a server is not useful.
One or more Providers must be named to give the server
a useful function.

Provider Interface
------------------

A Provider class will define some, or all, of the following:

.. class:: Provider

    .. method:: testChannel(pvname)

        Called with a PV name which some client is searching for.

        :return: True to claim this PV.

    .. method:: makeChannel(pvname, src):

        Called when a client attempts to create a Channel for some PV.
        The object which is returned will not be collected until
        the client closes the Channel or becomes disconnected.

        :return: An object which implements the Channel Interface, or None to disclaim this PV.

.. class:: Channel

    .. method:: rpc(response, request)

        Called each time a client issues a Remote Procedure Call
        operation on this Channel.

        :param response RPCReply: Use this to send the reply
        :param request Value: The raw arguments

Example RPC Provider
--------------------

See the :py:mod:`rpc` module for more functional RPC handling and dispatch.

>>> from p4p.nt import NTScalar
>>> class ExampleProvider(object):
        def __init__(self, myname):
            self.name = name
            # we 
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

    .. automethod:: start

    .. automethod:: stop

.. autofunction:: installProvider

.. autofunction:: removeProvider

.. autofunction:: clearProviders

.. autoclass:: RPCReply

    .. automethod:: done
