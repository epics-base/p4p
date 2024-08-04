.. _gwpage:

PVA Gateway
===========

.. currentmodule:: p4p

Quick Start
-----------

First install P4P (see the main :ref:`starting`).

The following commands will set up a gateway instance named ``mygw`` on a Linux system that uses *systemd*: ::

      # generate a simple configuration file
    sudo python -m p4p.gw --example-config /etc/pvagw/mygw.conf
      # generate a systemd unit file to support the gateway
    sudo python -m p4p.gw --example-systemd \
         /etc/systemd/system/pvagw@.service
      # start the gateway
    sudo systemctl daemon-reload
    sudo systemctl start pvagw@mygw.service
      # check to see if the instance has started correctly
    sudo systemctl status pvagw@mygw.service
      # set the instance to start automatically on boot
    sudo systemctl enable pvagw@mygw.service

Background
----------

The PVA Gateway provides a way for EPICS client software to access IOCs on an isolated network.

In doing so, it reduces the resource load on the server-facing side,
and provides access control restrictions to requests from the client facing side.
The gateway is a specialized proxy for the PV Access (PVA) Protocol
which sits between groups of PVA clients and servers.  (see `overviewpva`)

.. graph:: nogw
    :caption: PVA Connections without a Gateway

    rankdir="RL";
    serv1 [shape=box,label="EPICS IOC"];
    serv2 [shape=box,label="PVA server"];
    serv3 [shape=box,label="EPICS IOC"];
    cli1 [shape=box,label="pvget"];
    cli2 [shape=box,label="PVA client"];
    serv1 -- cli1
    serv2 -- cli1
    serv3 -- cli1
    serv1 -- cli2
    serv2 -- cli2
    serv3 -- cli2

Without a Gateway, ``M`` clients connect to ``N`` servers
with ``M*N`` connections (TCP sockets).  If all clients are subscribed
to the same set of PVs, then each server is sending the same data values
``M`` times.

.. graph:: gwnames
    :caption: PVA Connections through a Gateway

    rankdir="RL";
    serv1 [shape=box,label="EPICS IOC"];
    serv2 [shape=box,label="PVA Server"];
    serv3 [shape=box,label="EPICS IOC"];
    subgraph clustergw {
        label="Gateway\nProcess";
        gwc [label="Gateway\nClient"];
        gws [label="Gateway\nServer"];
    }
    cli1 [shape=box,label="pvget"];
    cli2 [shape=box,label="PVA Client"];

    serv1 -- gwc;
    serv2 -- gwc;
    serv3 -- gwc;
    gws -- cli1;
    gws -- cli2;

Adding a Gateway reduces the number of connections to ``M+N``.
With ``M`` clients connecting to a gateway server on one side, and one gateway client connecting to ``N`` servers on the other.
Further, a Gateway de-duplicates subscription data updates
so that each server sends only a single update to the Gateway,
which then repeats it to each client.

So the PVA servers and IOCs see only a single client,
and are shielded from a potentially large number of clients on
the other side of the gateway.

.. note::
    Each gateway process can define multiple internal Servers and Clients.
    This allows, for example, a single gateway process to connect to multiple IOC subnets,
    providing EPICS clients to access all IOCs.

Example
~~~~~~~

A common scenario is to have a gateway running on a host computer
with two network interfaces (NICs) on different subnets,
and thus two different broadcast domains.

In this example, a server has two NICs with IP addresses
192.168.1.5/24 and 10.1.1.4/24.

.. graph:: gwnet
    :caption: Example: A Multi-homed Host for a Gateway

    rankdir="LR";
    serv [shape=box,label="PVA server\n192.168.1.23"];
    cli  [shape=box,label="PVA client\n10.1.1.78"];
    subgraph clustergw {
        label="Gateway\nHost";
        nic2 [shape=cds,label="NIC 10.1.1.4",orientation=180];
        nic1 [shape=cds,label="NIC 192.168.1.5"];
    }
    cli -- nic2;
    nic1 -- serv;

To support this host, the gateway can be set up with the
following configuration file.
The intent is that the gateway provides EPICS clients on
the ``10.1.1.0/24`` subnet with access to IOCs or other PVA servers
on the ``192.168.1.0/24`` subnet.

Each of the statements in this configuration file are explained
below ::

    /* C-style comments are supported */
    {
        "version":2,
        "clients":[
            {
                "name":"client192",
                "addrlist":"192.168.1.255",
                "autoaddrlist":false
            }
        ],
        "servers":[
            {
                "name":"server10",
                "clients":["client192"],
                "interface":["10.1.1.4"],
                "addrlist":"10.1.1.255",
                "autoaddrlist":false,
                "statusprefix":"GW:STS:" /* optional, but suggested */
            }
        /* optional, allows server side access to Gateway status */
            ,{
                "name":"server192",
                "clients":[],
                "interface":["192.168.1.5"],
                "addrlist":"192.168.1.255",
                "autoaddrlist":false,
                "statusprefix":"GW:STS:"
            }
        ]
    }

The *version* statement is described below.

The *clients* section specifies the *name* of its only Client to be ``client192`` and is configured to search on the ``192.168.1.0/24`` subnet by
providing the ``192.168.1.255`` broadcast address as the only member of the *addrlist*.
This is the network to which an EPICS IOC is attached, so it will receive broadcast searches
from this gateway acting as a client.

The *servers* section specifies the *name* of its first Server to be ``server10``, and indicates which *clients* can have access to it, in this case clients which are part of the ``clients192`` section.
It is configured to listen on the ``10.1.1.0/24`` subnet by specifying the local *interface* address ``10.1.1.4``.
This is the network on which an EPICS client such as *pvget* or *pvput* is attached, and this gateway will act as a Server to receive their search messages.
The interface broadcast address is also provided to enable sending of server beacon packets.
This is an optimization to reduce connection time, and it is not required.

The *statusprefix* value is set to ``GW:STS:`` in this example, allowing the gateway to share some internal PVs which provide status information.
The :ref:`gwstatuspvs` suffixes are described below, with the *statusprefix* prepended.
Sites with multiple gateways on one subnet should give each a unique statusprefix.

A second *servers* section is shown, with its *name* set to ``server192``.  Its set of allowed *clients* is empty, but interfaces and address lists are specified.
This allows the status PVs mentioned above to be accessed from the subnet hosting the IOCs and other EPICS servers.
Without this section, those status PVs are only accessible from EPICS clients on the client subnets.

.. note::
    A single gateway will not connect to itself (no Gateway client will connect to a Gateway server in the same instance).
    However, this automatic loop avoidance is not possible in more complex situations involving multiple gateways.
    If such a setup is judged necessary, care should be taken to ensure that loops can not form.
    See also the ``servers[].ignoreaddr`` in :ref:`gwconfigfile`.

Command Line Arguments
----------------------

.. argparse::
    :module: p4p.gw
    :func: getargs
    :prog: pvagw

.. _gwconfref:

Configuration File
------------------

Configuration is provided as a file using JSON syntax with C-style comments. ::

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

See also PVXS client_ and server_ configuration references.

.. _client: https://epics-base.github.io/pvxs/client.html#configuration

.. _server: https://epics-base.github.io/pvxs/server.html#configuration

Run ``pvagw --example-config -`` to see another example configuration.

.. _gwconfigfile:

Configuration File Keywords
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a full list of JSON keys available for the configuration file, version 2.

**version**
    JSON Scheme version number.  2 is recommended for new files.  Valid values are 1 or 2.

**readOnly** (default: false)
    Boolean flag which, if set, acts as a global access control rule which rejects
    all PUT or RPC operations.  This takes precedence over any ACF file rules.

**clients**
    List of Gateway Client configurations.

**clients[].name**
    Unique name for this Client within this gateway process.

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
    List of gateway Server configurations.

**servers[].name**
    Unique name of this Server within this gateway process.

**servers[].clients**
    A list of zero or more gateway Client names.
    Search requests allowed through this server will be made through all listed clients.

**servers[].interface** (default: ["0.0.0.0"])
    A list of local interface addresses to which this gateway Server will bind.

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
    UDP port bound to receive search requests, as well as the port to which beacons are sent.

**servers[].getholdoff** (default: 0)
    A value greater than zero enables rate limiting of Get operations.
    ``getholdoff`` defines a hold-off time after a GET on a PV completes, before the another will be issued.
    Another GET for the same PV made before the hold-off expires will be delayed until expiration.
    Concurrent GET operations may be combined.

    This activity is per PV.

**servers[].statusprefix** (default: "")
    The text used by this gateway as a prefix to construct names for PVs which communicate status information.
    The PVs report overall status for the gateway process, regardless of the number of internal Clients or Servers.
    Each of the status PVs are defined in :ref:`gwstatuspvs`.
    Note that the prefix will typically end with the delimiter used in your PV naming convention, such as ``:``.

**servers[].access** (default: "")
    Name an ACF file to use for access control decisions for requests made through this server.
    See :ref:`gwacf`.
    Relative file names are interpreted in relation to the directory containing the config file.

**servers[].pvlist** (default: "")
    Name of PVList file used to restrict access to certain PVs through this Server.
    See :ref:`gwpvlist`.
    Relative file names are interpreted in relation to the directory containing the config file.

**servers[].acf_client**
    Needed only if ``access`` key is provided, and ``clients`` list has more than one entry.
    Unambiguously selects which client is used to connect ``INP`` PVs for use by conditional ACF rules.
    If not provided, then the first client in the list is used.

.. _gwstatuspvs:

Status PVs
----------

Servers with the ``statusprefix`` key set will provide access to the following PVs.
These values are aggregated from all defined internal gateway Servers and Clients.

.. warning::
    The PV names resulting from the ``statusprefix`` and the PV suffixes shown below must be unique across your site.
    Each gateway instance must have a unique ``statusprefix`` value.

**<statusprefix>asTest**
    An RPC only PV which allows testing of PVList and ACF rules. ::

        $ pvcall <statusprefix>asTest pv=some:name

    Other arguments include ``user="xx"``, ``peer="1.1.1.1:12345``, and ``roles=["yy"]``.
    If omitted, the credentials of the requesting client are used.

**<statusprefix>clients**
    A list of client's names connected to the GW server

**<statusprefix>cache**
    A list of channels to which the GW Client is connected

**<statusprefix>refs**
  Table of object type names and instance counts.
  May be useful for detecting resource leaks while troubleshooting.

**<statusprefix>threads**
  Available when running with python >= 3.5.
  An RPC call which returns a text description of all python threads.

.. note::
    The following PVs provide data bandwidth information for the overall gateway.

    * The ``ds`` in the names refer to *downstream* requests from EPICS clients to the gateway, or responses from the gateway to EPICS clients.
    * The ``us`` in the names refer to *upstream* requests from the gateway to IOCs, or responses from an IOC to the gateway.
    * The ``bypv`` or ``byhost`` in the names refer to status relating to the involved PVs or host machines, respectively.
    * The ``rx`` and ``tx`` in the names refer to receiving or transmitting data from the gateway's perspective.

**<statusprefix>ds:bypv:rx**
    A table containing bandwidth usage of requests for each PV sent from PVA clients such as **pvget** or **pvput** to this gateway.  This can be a relatively low number since the requests are often small in size.
    The table is sorted from highest bandwidth PVs to lowest.

**<statusprefix>us:bypv:tx**
    A table containing bandwidth usage of requests for each PV sent from this gateway to PVA Servers such as IOCs.  This can be a relatively low number since the requests are often small in size.
    The table is sorted from highest bandwidth PVs to lowest.

**<statusprefix>us:bypv:rx**
    A table containing bandwidth usage of responses from each PV sent from PVA Servers such as IOCs to this gateway.
    The table is sorted from highest bandwidth PVs to lowest.

**<statusprefix>ds:bypv:tx**
    A table containing bandwidth usage of responses from each PV sent from this gateway to EPICS clients that made the original requests.
    The table is sorted from highest bandwidth PVs to lowest.

**<statusprefix>ds:byhost:rx**
    A table containing bandwidth usage of each host sending requests from PVA clients such as **pvget** or **pvput** to this gateway.  This can be a relatively low number since the requests are often small in size.
    The table is sorted by host machine with the highest bandwidth usage to lowest.

**<statusprefix>us:byhost:tx**
    A table containing bandwidth usage of requests sent from this gateway to each host containing PVA Servers such as IOCs.  This can be a relatively low number since the requests are often small in size.
    The table is sorted by host machine with the highest bandwidth usage to lowest.

**<statusprefix>us:byhost:rx**
    A table containing bandwidth usage of each host providing responses from PVA Servers such as IOCs to this gateway.
    The table is sorted by host machine with the highest bandwidth usage to lowest.

**<statusprefix>ds:byhost:tx**
    A table containing bandwidth usage of each client's host accepting responses from this gateway.
    The table is sorted by host machine with the highest bandwidth usage to lowest.

.. _gwlogconfig:

Log File Configuration
----------------------

The gateway is able to record messages associated with important events to one or more destinations as it runs,
including log files or a console device.
The messages can be debugging aids for developers,
or errors encountered as the gateway is working.
It also records the time at which the gateway starts or stops, and when starting,
lists the configuration details for the internal clients and servers, and lists each status PV that the gateway will make available.

A python `dictConfig <https://docs.python.org/3/library/logging.config.html#logging-config-dictschema>`_
logging configuration file in JSON format can be passed to ``--logging``
to provide control of formating of the logged messages.

One logger name of special interest is ``p4p.gw.audit`` which is used for messages arising from :ref:`trapwrite`.

The following is an example of a log configuration file which records ``INFO`` messages or worse to a log file,
but also records ``WARNING`` messages or worse to the computer console.
It specifies different formats for console-bound messages versus log file messages,
and instructs the system to maintain daily log files (and audit files, if enabled),
in a subdirectory called ``BL3-LOGS``.
It will create new, empty log files each midnight while keeping previous log files for 14 days.

Note that fixed-width columns are specified for some fields using sequences like ``15s``, ``-4d`` or ``4.4s``,
similar to ``printf`` style format specifiers:  ::

        {
            "version": 1,
            "disable_existing_loggers": false,
            "formatters": {
                "fileFormat": {
                    "format": "%(asctime)s | %(name)15s line %(lineno)-4d [%(levelname)4.4s] %(message)s"
                },
                "consoleFormat": {
                    "format": "%(asctime)s | %(name)s: %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "fileMessages": {
                    "level": "INFO",
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "formatter": "fileFormat",
                    "filename": "BL3-LOGS/gateway-BL3-DMZ.log",
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": 14
                },
                "consoleMessages": {
                    "level": "WARNING",
                    "class": "logging.StreamHandler",
                    "formatter": "consoleFormat",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["fileMessages","consoleMessages"],
                    "level": "INFO",
                    "propagate": true
                }
            }
        }

.. _gwsec:

Access Control Model
--------------------

A gateway may apply access control restrictions in addition to any restrictions
applied by individual IOCs, or other PVA servers, to which it connects.
By default a gateway apply **no restrictions**.
A gateway without a :ref:`gwpvlist` or :ref:`gwacf` will allow all clients to attempt any operation on any PV.

One or more of the ``readOnly``, ``access``, and/or ``pvlist`` configuration file keys enable
restrictions within a gateway.

The **simplest and most direct restriction is the readOnly** configuration file key,
which applies to all logical Servers within a gateway.
If set, no PUT or RPC operations are allowed.
Both MONITOR and GET operations are allowed, so ``readOnly`` applies a simple one-way policy
to allow clients to receive data without being permitted to change any PV settings.

A more granular policy is often desired,
which can be expressed with a :ref:`gwpvlist` and/or :ref:`gwacf`.

Access decisions are made as follows:

1. PV name and client IP address are looked up in a PVList.
   If DENY, then client searches are ignored.
2. If ``readOnly`` is set, then any PUT or RPC operation is rejected.  GET/MONITOR proceed.
3. The PV name and peer IP address are looked up in a PVList.
   If DENY, then the operation is rejected.
   If ALLOW/ALIAS then an ASG name and level (0 or 1) is found.
4. The ASG name and level are look up in a ACF file.
   GET/MONITOR operations are always allowed (no write only PVs).
   PUT or RPC operations are allowed if appropriate WRITE/PUT/RPC permission is granted.

.. _gwpvlist:

PVList File
-----------

The purpose of the PVList file is to specify which PVs are allowed or denied,
and to associate those PVs with access security groups (ASG) and security levels (ASL) in the access file.
Supported PVList file syntax is mostly compatible with that of the Channel Access Gateway_.

.. _Gateway: https://epics.anl.gov/EpicsDocumentation/ExtensionsManuals/Gateway/Gateway.html

If not provided, the default PVList file used is ``.* ALLOW``.

While allows all PV names from all clients.

PVList files are line based, with lines consisting of the following.

.. note:: Matching is *not* in strict lexical order.  See below.

#. Blank lines and ``#`` comments are allowed. ::

    # comment line

#. **Evaluation order statement**, primarily to maintain backward compatibility with CA Gateway. ::

    # (default if omitted)
    EVALUATION ORDER ALLOW, DENY

    # Not currently supported
    #EVALUATION ORDER DENY, ALLOW

#. **A DENY statement** which specifies that certain PVs are denied access from certain EPICS clients.
   It can specify an optional host from which clients will be denied access.
   ``<PV name regexp>`` is a `regular expression`_ to match PV names.
   This statement is of the forms ::

   <PV name regexp> DENY
   <PV name regexp> DENY FROM <hostname_or_IP>

.. _regular expression: https://docs.python.org/3/library/re.html#regular-expression-syntax

#. **An ALLOW statement** which specifies that certain PVs are allowed to be accessed from EPICS clients.
   It can specify an optional *Access Security Group* (ASG),
   with an accompanying but optional *Access Security Level* (``0`` or ``1``),
   both of which used when evaluating an **ACF** file.
   This statement is of the forms ::

   <PV name regexp> ALLOW
   <PV name regexp> ALLOW <ASG>
   <PV name regexp> ALLOW <ASG> <ASL_0_or_1>

If not provided, ``ASG`` is ``DEFAULT``, and ``ASL`` is ``0``.

#. **An Alias statement** which provides a way to specify a specific PV name based on a more general pattern.
   This is equivalent to a **ALLOW** statement with an additional name translation.
   This statement is of the forms ::

   <PV name regexp> ALIAS <real PV name>
   <PV name regexp> ALIAS <real PV name> <ASG>
   <PV name regexp> ALIAS <real PV name> <ASG> <ASL_0_or_1>

When a gateway Server receives a request from a client to access a PV,
the PV name is compared to each pattern in the list.

The order in which regular expressions are matched is that all DENY statements are considered
before any ALLOW/ALIAS statements (regardless of the order of lines).
PV names which do not match any statement are DENYed.

When a PV name matches more than one ALLOW/ALIAS statements,
the order of lines is used.
The last match will have effect.

Considering the following PVList file: ::

    ACCL:CRYO:.* DENY
    ACCL:.*      ALLOW MISCONFIG
    ACCL:.*      ALLOW
    ACCL:RF.*    ALLOW RF

``ACCL:CRYO:ESTOP`` would match the ``DENY`` rule, so a gateway will not allow any access.

``ACCL:RF:FPWR`` would match the ``ALLOW RF`` rule, and be allowed subject to rules for ``ASG(RF)``.

``ACCL:ARC:CNT`` would match the last ``ALLOW`` rule, and be allowed subject to ``ASG(DEFAULT)``.

Because both the ``ALLOW MISCONFIG`` and ``ALLOW`` rules have identical patterns,
the ``ALLOW`` will always be used and ``ALLOW MISCONFIG`` will never be used.

Note that because ``DENY`` rules are always considered before ``ALLOW`` or ``ALIAS`` rules,
the preceding file is functionally identical to the following as moving the ``DENY``
relative to ``ALLOW`` does not change the evaluation order.  ::

    ACCL:.*      ALLOW MISCONFIG
    ACCL:.*      ALLOW
    ACCL:RF.*    ALLOW RF
    ACCL:CRYO:.* DENY

When building a PVList file containing ``ALLOW`` or ``ALIAS`` rules with overlapping patterns,
it is therefore necessary to put the more general patterns before the more specific patterns.
eg. ::

    ACCL:RF.*    ALLOW RF
    ACCL:.*      ALLOW

In this example the ``ALLOW RF`` rule is effectively hidden, and will never be matched.


.. _gwacf:

ACF Rules File
--------------

An Access Security File (ACF) is a list of access control rules to be applied
to requests based on which Access Security Group (``ASG``) was selected by a PVList file, or ``DEFAULT``
if no PVList file is used.  The ``ASG`` name selects which a group of rules.

Unknown ``ASG`` names use the ``DEFAULT`` rules.
If no ``DEFAULT`` group is defined, then no privileges are granted.

Each ACF file may define zero or more Host Access Groups (``HAG`` s) and/or
User Access Groups (``UAG`` s).
Also, one or more list of rules (``ASG`` s).
The HAG is a list of host names, and the UAG a list of user names.

eg. PVs in ASG ``DEFAULT`` only permit PUT or RPC requests originating from
hosts ``incontrol`` or ``physics``.
PUT requests from ``physics`` will be logged. ::

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
    Shorthand to grant both ``PUT`` and ``RPC`` requests.

``PUT``
    Allow PUT operation on all fields.

``RPC``
    Allow RPC operations.

``UNCACHED``
    Special privilege which allows a client to bypass deduplication/sharing of subscription data.
    A client would make use of this privilege by including a pvRequest option ``record._options.cache``
    with a boolean false value.

``READ``
    Accepted for compatibility.
    PVA Gateway always allows read access for any PV which is allowed by the PVList file.
    Use a ``DENY`` in a PVList file to prevent clients from reading or subscribing to certain PVs.

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
(See ``/etc/nsswitch.conf`` on Linux).

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

.. _trapwrite:

TRAPWRITE and Put logging
~~~~~~~~~~~~~~~~~~~~~~~~~

If a ``RULE`` includes the ``TRAPWRITE`` modifier, then a ``PUT`` operation it allows
will be logged.
Refer to the :ref:`gwlogconfig` section for more information.

Messages are logged through the ``p4p.gw.audit`` python logger.

ACF Syntax
~~~~~~~~~~

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

Application Notes
-----------------

The process of configuring a Gateway begins by looking at the
physical and/or logical topology of the networks in question.

A Gateway is typically placed at the boundary between one or more networks (subnets).

While a simple Gateway configuration will have a single GW Server connected to a single GW Client,
more complicated configurations are possible, with many GW Servers and one GW Client,
or one GW Server and many GW Clients, or a many to many configuration.

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
In the interest of performance, Python code is only in the "slow" path
for the PV search/connection decision.
After a PV is connected; permissions changes, auditing, and monitoring are communicated
externally from Python code.

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

1. Create a `ClientProvider`
2. Create a `Provider` using this client
3. Create a `p4p.server.Server` referencing the provider name.

More than one `Provider` may reference to the same `ClientProvider`.
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

    .. automethod:: sweep

    .. automethod:: forceBan

    .. automethod:: clearBan

    .. automethod:: cachePeek

    .. automethod:: stats

    .. automethod:: report

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
