.. _gwpage:

PVA Gateway
===========

.. currentmodule:: p4p

The PVA Gateway is a specialized proxy for the PV Access (PVA) Protocol
which sits between groups of PVA client and of servers.  (see `overviewpva`)
It serves two broad roles.
To reduce the resource load on the server-facing side,
and to apply access control restrictions to requests from the client facing side.

.. graph:: nogw
    :caption: Connections without Gateway
    
    rankdir="LR";
    serv1 [shape=box,label="PVA Server"];
    serv2 [shape=box,label="PVA Server"];
    serv3 [shape=box,label="PVA Server"];
    cli1 [shape=box,label="PVA Client"];
    cli2 [shape=box,label="PVA Client"];
    serv1 -- cli1
    serv2 -- cli1
    serv3 -- cli1
    serv1 -- cli2
    serv2 -- cli2
    serv3 -- cli2

In this situation without a Gateway ``M`` clients connect to ``N`` servers
with ``M*N`` TCP connections (sockets).  If all clients are subscribed
to the same set of PVs, then each server is sending the same data values
``M`` times.

.. graph:: gwnames
    :caption: Gateway processes and connection

    rankdir="LR";
    serv1 [shape=box,label="PVA Server"];
    serv2 [shape=box,label="PVA Server"];
    serv3 [shape=box,label="PVA Server"];
    subgraph clustergw {
        label="GW Process";
        gwc [label="GW Client"];
        gws [label="GW Server"];
    }
    cli1 [shape=box,label="PVA Client"];
    cli2 [shape=box,label="PVA Client"];

    serv1 -- gwc;
    serv2 -- gwc;
    serv3 -- gwc;
    gws -- cli1;
    gws -- cli2;

Adding a Gateway reduces the number of connections to ``M+N``.
With ``M`` one side, and ``N`` on the other.
Further, a Gateway de-duplicates subscription data updates
so that each server sends only a single copy to the Gateway,
which then repeats it to each client.

These two facts combine to shield the Servers from an excessive
numbers of Clients.

A prototypical scenario of Gateway usage is on a host computer
with two network interfaces (NICs) on different subnets
(and thus two different broadcast domains).

To take an example.  A server has two NICs with IP addresses
192.168.1.5/24 and 10.1.1.4/24 .

.. graph:: gwnet

    rankdir="LR";
    serv [shape=box,label="PVA Server\n192.168.1.23"];
    cli  [shape=box,label="PVA Client\n10.1.1.78"];
    net1 [shape=none,label="Net 192.168.1.0/24"];
    net2 [shape=none,label="Net 10.1.1.0/24"];
    subgraph clustergw {
        label="GW Host";
        nic1 [shape=cds,label="NIC 192.168.1.5"];
        nic2 [shape=cds,label="NIC 10.1.1.4"];
    }
    net1 -- nic1;
    net1 -- serv;
    net2 -- nic2;
    net2 -- cli;

In the following configuration we wish a client running
on the host ``10.1.1.78`` to be able to
communicate with a server running on ``192.168.1.23``.  ::

    /* JSON with C-style comments */
    {
        "version":2,
        "clients":[
            {
                "name":"client192",
                "addrlist":"192.168.1.255",
                "autoaddrlist":false,
            }
        ],
        "servers":[
            {
                "name":"server10",
                "clients":["client192"],
                "interface":["10.1.1.4"],
                "addrlist":"10.1.1.255",
                "autoaddrlist":false,
                "statusprefix":"GW:STS:", /* optional, but suggested */
            }
            /* optional, allows server side access to Gateway status */
            ,{
                "name":"server192",
                "clients":[],
                "interface":["192.168.1.5"],
                "addrlist":"192.168.1.255",
                "autoaddrlist":false,
                "statusprefix":"GW:STS:",
            }
        ]
    }

GW Client ``client192`` is configured to search on the ``192.168.1.0/24`` subnet by
providing the ``192.168.1.255`` broadcast address.  This is the network
to which the PVA Server is attached, so it will receive broadcast searches
from this GW Client.

GW Server ``server10`` is configured to listen on the ``10.1.1.0/24`` subnet by providing
the local interface address ``10.1.1.4``.  This is the network to which the PVA
Client is attached, so this GW Server will receive search messages sent
by the client.
The interface broadcast address is also provided to enable sending of server
beacon packets.  This is an optimization to reduce connection time, and not required.

Additionally, both GW Servers ``server10`` and ``server192`` are configured to provide internal Gateway status
PVs with the name prefix ``GW:STS:``.  See `gwstatuspvs` for details.

This Gateway may be started by saving the preceding JSON as a file ``mygw.conf`` ::

    pvagw mygw.conf

CLI Arguments
-------------

.. note::
    Unless the ``--no-ban-local`` argument is passed, a Gateway
    will ignore all Client connection attempts originating from
    the same host.  This prevents a mis-configured Gateway from
    connecting to itself, but may cause surprise during gateway
    setup and testing.

.. argparse::
    :module: p4p.gw
    :func: getargs
    :prog: pvagw

.. _gwconfref:

Configuration File
------------------

Configuration is provided as a file using JSON syntax with C-style comments.
A full list of known keys for configuration scheme version 2. ::

    /* C-style comments allowed */
    {
        "version":2,
        "readOnly":false,
        "clients":[
            {
                "name":"theclient",
                "provider":"pva",
                "addrlist":"...",
                "autoaddrlist":false,
                "bcastport":5076
            }
        ],
        "servers":[
            {
                "name":"theserver",
                "clients":["theclient"],
                "interface":["..."],
                "addrlist":"",
                "ignoreaddr":["..."],
                "autoaddrlist":false,
                "serverport":5075,
                "bcastport":5076,
                "getholdoff":1.0,
                "statusprefix":"PV:",
                "access":"somefilename.acf",
                "pvlist":"somefilename.pvlist"
            }
        ]
    }

Keys
~~~~

**version**
    Scheme version number.  2 is recommended for new files.  Valid values are 1 or 2.

**readOnly** (default: false)
    Boolean flag which, if set, acts as a global access control rule which rejects
    all PUT or RPC operations.  This take precedence over any ACF file rules.

**clients**
    List of GW Client configurations.

**clients[].name**
    Unique name for this GW Client

**clients[].provider** (default: "pva")
    Selects a ChannelProvider.  Currently only "pva" is valid.

**clients[].addrlist** (default: "")
    List of broadcast and unicast addresses to which search messages will be sent.

**clients[].autoaddrlist** (default: true)
    Whether to automatically populate *addrlist* with **all** local interface broadcast addresses.
    Use caution when setting ``true``.

**clients[].bcastport** (default: 5076)
    UDP port to which searches are sent.

**servers**
    List of GW Server configurations.

**servers[].name**
    Unique name of this GW Server

**servers[].clients**
    A list of zero or more GW Client names.  Search requests allowed through this server
    will be made through all listed clients.

**servers[].interface** (default: ["0.0.0.0"])
    A list of local interface addresses to which this GW Server will bind.

**servers[].addrlist** (default: "")
    List of broadcast and unicast addresses to which beacon messages will be sent

**servers[].ignoreaddr** (default: "")
    List of address to add into the banned list to explicit ignore hosts.

**servers[].autoaddrlist** (default: true)
    Whether to automatically populate *addrlist* with **all** local interface broadcast addresses.
    Use caution when setting ``true``.

**servers[].serverport** (default: 5075)
    Default TCP port to bind.  If not possible, a random port will be used.

**servers[].bcastport** (default: 5076)
    UDP port bound to receive search requests.  Also to which beacons are sent.

**servers[].getholdoff** (default: 0)
    A value greater than zero enables rate limiting of Get operations.  ``getholdoff`` defines as a hold-off time
    after a GET on a PV completes before the another will be issued.  Another GET for the same PV
    made before the hold-off expires will be delayed until expiration.
    Concurrent GET operations may be combined.

    This activity is per PV.

**servers[].access** (default: "")
    Name an ACF file to use for access control decisions for requests made through this server.
    See `gwacf`.
    Relative file names are interpreted in relation to the directory containing the config file.

**servers[].pvlist** (default: "")
    Name of PV List file to use for access control decisions for PVs accessed through this server.
    See `gwpvlist`.
    Relative file names are interpreted in relation to the directory containing the config file.

**servers[].acf_client**
    Needed only if ``access`` key is provided, and ``clients`` list has more than one entry.
    Unambiguously selects which client is used to connect ``INP`` PVs for use by conditional ACF rules.
    If not provided, then the first client in the list is used.

.. _gwstatuspvs:

Status PVs
----------

Servers with the ``statusprefix`` key set will provide access to the following PVs.
These values are aggregated from all GW Servers and GW Clients.

**<statusprefix>asTest**
    An RPC only PV which allows testing of pvlist and ACF rules. ::

        $ pvcall <statusprefix>asTest pv=some:name

    Other arguments include ``user="xx"``, ``peer="1.1.1.1:12345``, and ``roles=["yy"]``.
    If omitted, the credentials of the requesting client are used.

**<statusprefix>clients**
    A list of clients names connected to the GW server

**<statusprefix>cache**
  A list of channels to which the GW Client is connected

**<statusprefix>us:bypv:tx**

**<statusprefix>us:bypv:rx**

**<statusprefix>ds:bypv:tx**

**<statusprefix>ds:bypv:rx**

**<statusprefix>us:byhost:tx**

**<statusprefix>us:byhost:rx**

**<statusprefix>ds:byhost:tx**

**<statusprefix>ds:byhost:rx**
  Each is a table showing bandwidth usage reports aggregated in various ways.

  ``us`` for upstream, GW Client side.  ``ds`` for downstream, GW Server side.

  ``bypv`` vs. ``byhost`` group results by the PV name involved, or the peer host.
  ``us:byhost:*`` is grouped by upstream server (IOC).  ``ds:byhost:*`` is grouped
  by downstream client.

  ``tx`` vs. ``rx`` is direction of data flow as seen by the gateway process.

  eg. ``us:byhost:rx`` is data received from Servers by the GW Client grouped
  by Server IP.

  eg. ``ds:bypv:tx`` is data send by the GW Server to Clients grouped by PV name.

.. _gwsec:

Access Control Model
--------------------

A Gateway can enforce access control restrictions on requests flowing through it.
However, **no restrictions** are made by default.
And a Gateway will attempt to connect any PV and allow any operation.
One or more of the ``readOnly``, ``access``, and/or ``pvlist`` configuration file
keys is needed to enable restrictions.

The simplest and more direct restriction is the ``readOnly`` configuration file key.
If set, no PUT and RPC operations are allowed.
MONITOR and GET operations are allowed, so ``readOnly`` applies a simple one-way policy
to allow Clients to receive data without being permitted to change settings.

A more granular policy may be defined in separate PV List file and/or ACF file.

A combination of PV List and ACF may take as into consideration the PV name being searched
and the Client host name/IP when deciding whether to allow a PV.
Further, allowed PVs then provide credentials which may be used to grant specific privileges
needed for some operations (mainly PUT and RPC).

.. _gwpvlist:

PV List File
------------

A PV List file contains a list of regular expressions, each with a corresponding
action.  Either to deny (ignore) a Client search attempt, or to allow it through,
possibly with a different PV name, and/or subject to
further restrictions in an ACF file (according to ASG name).

Supported PV List file syntax is mostly compatible with that of the Channel Access Gateway.
At present, only the "ALLOW, DENY" evaluation order is supported.

If no PV List file is provided, an implicit default is used
which allows any PV name through under the ``DEFAULT`` ASG. ::

    # implied default PV List file
    .* ALLOW DEFAULT 1

Syntax is line based.  Order of precedence is **DENY over ALLOW** and **last to first**.

So a line ``.* DENY`` intended to block all names not specifically allowed
must be placed at the top of the file.

Valid (non-blank) lines are ::

    # comment

    # explicitly specify evaluation order.
    # ALLOW, DENY is the default
    # DENY, ALLOW is not supported
    EVALUATION ORDER ALLOW, DENY

    # Allow matching PVs.  Use ASG DEFAULT and ASL 1
    <regexp> ALLOW
    # Allow matching PVs.  Use ASG MYGRP and ASL 1
    <regexp> ALLOW MYGRP
    # Allow matching PVs.  Use ASG MYGRP and ASL 0
    <regexp> ALLOW MYGRP 0

    # Allow Client requests matching PVs.  Forward to Servers under a different name
    # regexp captures may be used.
    # otherwise behaves like ALLOW
    <regexp> ALIAS <subst>
    <regexp> ALIAS <subst> MYGRP
    <regexp> ALIAS <subst> MYGRP 0

    # Ignore any client searches
    <regexp> DENY

    # Ignore specific searches from a specific client
    <regexp> DENY FROM <hostname>

.. _gwacf:

ACF Rules File
--------------

An Access Security File (ACF) is a list of access control rules to be applied
to requests based on which ASG was selected by a PV List file, or ``DEFAULT``
if no PV List file is used.  The ``ASG`` name selects which a group of rules.

Unknown ``ASG`` names use the ``DEFAULT`` rules.
If no ``DEFAULT`` group is defined, then no privileges are granted.

Each ACF file may define zero or more groups of host names (``HAG`` s) and/or
user names (``UAG`` s).  Also, one or more list of rules (``ASG`` s).

Syntax
~~~~~~

.. productionlist:: acf
    acf: | item acf
    item : uag | hag | asg
    uag : UAG ( "NAME" ) { users }
    hag : HAG ( "NAME" ) { hosts }
    asg : ASG ( "NAME" ) { asitems }
    users : "HOSTNAME"
          :"HOSTNAME" , users
    hosts : "USERNAME"
          : "USERNAME" , hosts
    asitems : | asitem asitems
    asitem : INP[A-Z] ( "PVNAME" )
           : RULE ( ASL#, priv) rule_cond
           : RULE ( ASL#, priv, trap) rule_cond
    priv : READ | WRITE | PUT | RPC | UNCACHED
    trap : TRAPWRITE | NOTRAPWRITE
    rule_cond : | { conds }
    conds : | cond conds
    cond : UAG ( "NAME" )
         : HAG ( "NAME" )
         : CALC ( "EXPR" )

eg. PVs in ASG ``DEFAULT`` only permit PUT or RPC requests originating from
hosts ``incontrol`` or ``physics``.  PUT requests from ``physics`` will logged. ::

    HAG(MCF) { "incontrol" }
    HAG(OTHER) { "physics" }
    ASG(DEFAULT) {
        RULE(1, WRITE) {
            HAG(MCF)
        }
        RULE(1, WRITE, TRAPWRITE) {
            HAG(OTHER)
        }
    }

Privileges
~~~~~~~~~~

``RULE`` s may grant one of the following privileges.

``WRITE``
    Shorthand to grant both ``PUT`` and ``RPC``.

``PUT``
    Allow PUT operation on all fields.

``RPC``
    Allow RPC operation

``UNCACHED``
    Special privilege which allows a client to bypass deduplication/sharing of subscription data.
    A client would make use of this privilege by including a pvRequest option ``record._options.cache``
    with a boolean false value.

``READ``
    Accepted for compatibility.
    PVA Gateway always allows read access for any PV which is allowed by the PV List file.
    Use a ``DENY`` in a PV List file to prevent client(s) from reading/subscribing to certain PVs.

HAG Hostnames and IPs
~~~~~~~~~~~~~~~~~~~~~

Entries in a ``HAG()`` may be either host names, or numeric IP addresses.
Host names are resolved once on Gateway startup.
Therefore, changes in the hostname to IP mapping will not be visible
until a Gateway is restarted.

.. _gwpvcred:

UAG and Credentials
~~~~~~~~~~~~~~~~~~~

PV Access protocol provides a weakly authenticated means of identification based on a remotely provided user name.
This is combined with a set of "role"s taken by looking up system groups of which the username is a member.
(See ``/etc/nsswitch.conf``).

Both user and role names may appear in ``UAG`` lists. eg. ::

    UAG(SPECIAL)
    {
        root,
        "role/admin"
    }

And a rule: ::

    ASG(DEFAULT) {
        RULE(1, WRITE) {
            UAG(SPECIAL)
        }
    }

In this case, the ``RULE`` will be match if a client identifies itself with username ``root``
or if the (remotely provided) username is a member of the (locally tested) ``admin`` role (eg. unix group).

In this case, such a match will grant the ``WRITE`` privilege for PVs in the ``DEFAULT`` ASG.

Role/group membership can be tested with the ``<statusprefix>asTest`` status PV.

TRAPWRITE and Put logging
~~~~~~~~~~~~~~~~~~~~~~~~~

If a ``RULE`` includes the ``TRAPWRITE`` modifier, then a ``PUT`` operation it allows
will be logged through the ``p4p.gw.audit`` python logger.

See the ``--logging`` CLI argument,
and the python documentation of `dictConfig() <https://docs.python.org/library/logging.config.html#logging.config.dictConfig>`_

Application Notes
-----------------

The process of configuring a Gateway will usually begin by looking at the
physical and/or logical topology of the networks in question.

A Gateway is typically placed at the boundary between one or more networks (subnets).

While a simple Gateway configuration will have a single GW Server connected to a single GW Client,
more complicated configurations are possible, with many GW Servers and one GW Client,
on GW Server and many GW Clients, or a many to many configuration.

It is valid for a GW Client and GW Server to be associated with the same host interface and port
provided that they are not associated with each other.
Pairs of such GW Client and GW Server may be cross linked to form a bi-directional Gateway.

It is meaningful to configure a GW Server with no GW Clients ( ``"clients":[]`` )
provided that the ``"statusprefix"`` key is set.
This server will only provide the status PVs.
This can be used to eg. provide GW status monitoring from both sides of a one-way Gateway.


Differences from CA gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Summary of known differences from CA gateway.

* ``EVALUATION ORDER DENY, ALLOW`` is not supported.

* Permission ``READ`` is implied.  Write-only PVs are not possible.

Implementation Details
----------------------

Gateway is implemented as a hybrid of Python and C++.
In the interest of performance, Python code is only in the "fast" path
for the PV search/connection decision.
After a PV is connected; permissions changes, auditing, and monitoring are communicated
asynchronously from Python code.

The APIs described below are not currently considered stable or public for use by external modules.
They are documentation here for the purposes of internal development and debugging.

Negative Results Cache
~~~~~~~~~~~~~~~~~~~~~~

In order to shield the Python testChannel() handler from repeated reconnect attempts
for denied PVs, a list of blocked PVs, IPs, and pairs of PV and IP
is maintained in C++ code.  Search requests matching one of these three criteria
will be ignored without calling testChannel().

p4p.gw Frontend
~~~~~~~~~~~~~~~

This module utilizes the related C++ extension to setup and manage a Gateway
which is configured in a manner similar to the `pva2pva gateway <https://epics-base.github.io/pva2pva/>`_
with an access control policy defined in a manner similar to `cagateway <https://epics.anl.gov/extensions/gateway/>`_.
Other means of configuration and policy definition could be implemented.

C++ Extension
~~~~~~~~~~~~~

Setup execution flow for use of the C++ extension is:

1. Create a `Client`
2. Create a `Provider` using this client
3. Create a `p4p.server.Server` referencing the provider name.

More than one `Provider` may reference to the same `Client`.
A `p4p.server.Server` may reference more than one `Provider`,
and a `Provider` may be referenced by more than one `p4p.server.Server`.
Many `p4p.server.Server` s may be created.

After server startup, the handler object associated with a `Provider`
will be called according to the `_gw.ProviderHandler` interface.

The C++ extension deals only with IP addresses in string form,
possibly with port number (eg. "1.2.3.4:5076)", and never host names.

.. module:: p4p._gw

.. autoclass:: Provider

    .. autoattribute:: Claim

    .. autoattribute:: Ignore

    .. autoattribute:: BanHost

    .. autoattribute:: BanPV

    .. autoattribute:: BanHostPV

    .. automethod:: testChannel

    .. automethod:: disconnect

    .. automethod:: sweep

    .. automethod:: forceBan

    .. automethod:: clearBan

    .. automethod:: cachePeek

    .. automethod:: stats

    .. automethod:: report

.. autoclass:: Client

.. autoclass:: InfoBase
    :members:
    :undoc-members:

.. autoclass:: CreateOp
    :members:
    :undoc-members:

.. autoclass:: Channel
    :members:
    :undoc-members:

Interfaces
~~~~~~~~~~

.. class:: ProviderHandler

    A Handler object associated with a `Provider` should implement these methods

    .. method:: testChannel(self, pvname, peer)

        :param str pvname: PV name being searched (downstream)
        :param str peer: IP address of client which is searching
        :returns: Claim, Ignore, BanHost, BanPV, or BanHostPV

        Hook into search phase.  Called each time a client searches for a pvname.
        If permitted, call and return the result of `Provider.testChannel()` with the desired upstream (server-side PV name).

        * Returning Claim may result in a later call to `makeChannel()`.
        * Returning Ignore may result in a repeated call to testChannel() in future.
        * Returning BanHost adds this host to the negative results cache
        * Returning BanPV adds this PV to the negative results cache.
        * Returning BanHostPV adds this combination of host and PV to the negative results cache

    .. method:: makeChannel(self, op)

        Hook info channel creation phase.  If permitted, call and return the result of `CreateOp.create()`.
        The `Channel` object may be stored by python code to track and effect active connections.
        eg. call `Channel.access()` to set/change privileges.
        Or `Channel.close()` to force disconnection.

        Due to the continuous nature of PVA client (re)connection process, inability to create
        a channel at this stage is treated as a hard failure to avoid a reset loop.
        If it is necessary to return None, then steps should be taken to ensure that a
        re-connection attempt would have a different result.  eg. through `Provider.forceBan()`.

        :param CreateOp op: Handle for ongoing operation
        :returns: A `Channel`.

    .. method:: audit(self, msg)

        Hook info PUT logging process.  Called from a worker thread.

        :param str msg: Message string to be logged
