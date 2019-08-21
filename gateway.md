# PVA Gateway

## Terminology

The following discussion references four parts
of a Channel connected through a PV Access Gateway.

Downstream Client -> Gateway Server -> Gateway Client -> Upstream Server

Gateway Server/Client are the two "ends" of the gateway process,
which might be acting on eg. two different host network interfaces.

## Running

```sh
$ PYTHONPATH=... python -m p4p.gw -h
usage: gw.py [-h] [--no-ban-local] [-v] [--logging LOGGING] [--debug] config

positional arguments:
  config             Config file

optional arguments:
  -h, --help         show this help message and exit
  --no-ban-local     Skip ban of local interfaces, which prevents local
                     clients. Allow GW to talk to itself.
  -v, --verbose
  --logging LOGGING  Use logging config from file (JSON in dictConfig format)
  --debug
```

## Configuration file structure

The configuration file is JSON with C-style /* */ comments.
The basic structure is:

```json
{
    "version":1,
    "readOnly":false, /* optional */
    "clients":[...], /* Gateway Client instances */
    "servers":[...]  /* Gateway Server instances */
}
```

The configuration version number must be 1.

The optional "readOnly" flag defaults to 'false' if omitted.
If true, then no Put or RPC operations are possible through the gateway.

The "clients" section section contains a list of Gateway Client configurations
which may be referenced from one or more "server" entries (Gateway Servers).
Each "server" entry specifies an interface, or interfaces, to which
a PVA server will be bound.

### Client configuration

```json
{
    ...
    "clients":[
        {
            "name":"aclient",    /* required */
            /* "provider":"pva",  */
            /* "serverport":5075, */
            /* "bcastport":5076,  */
            "addrlist":"192.168.210.1 192.168.210.255",
            "autoaddrlist":false
        }
    ]
}
```

The primary client configuration is of "addrlist" and "autoaddrlist".
Most gateway configurations will specify "autoaddrlist" as 'false'.
However, for consistency the default remains 'true'.

"addrlist" may contain a mix of local interface broadcast and unicast addresses.
Any hostnames given are resolved to IP addresses on startup.

### Server configuration

```json
{
    ...
    "servers":[
        {
            "name":"aserver",           /* required */
            "clients":["aclient"],
            "interface":["192.168.210.1"],
            "addrlist":"",              /* beacon address list */
            "autoaddrlist":true,
            /* "serverport":5075, */
            /* "bcastport":5076,  */
            "getholdoff":1.0,           /* optional */
            "statusprefix":"sts:",      /* optional.  eg. allows "sts:clients" */
            "pvlist":"mypvs.acf",       /* optional */
            "access":"myrules.acf"      /* optional */
        }
    ]
}
```

Each server entry specifies one or more local interface address,
or the (default) wildcard 0.0.0.0 address.

It references zero or more client sections.
A Gateway Server will attempt to connect channels through any associated Gateway Client.
Additionally, a server "statusprefix" may be specified, to expose
a set of internal status PVs.  These are described below.

Each server may also reference "pvlist" and/or "access" (ACF) files
to govern access control decisions for downstream clients attempting
to connect through it.

If no "pvlist" file is provided, then all search requests are allowed through.

If no "access" files is provided, then any connected channels are
writable (subject to the global "readOnly" flag).

If only a "pvlist" file is provided, then any DENY and ALIAS entries are honored.
Allowance of Put and RPC is subject to the "readOnly" flag.

If only a "access" file is provided, then only ASG DEFAULT will be used.
Allowance of Put and RPC is subject to the "readOnly" flag and the DEFAULT RULEs.

The "getholdoff" key, if present with a value greater than zero enables
rate limiting of Get operations.  This limit is defined as a hold-off time
after a Get on a PV completes before the another can be issued.
This timer is per PV.

Downstream client Get operations during the hold-off period are queued
until the period ends, at which point they are forwarded.

## Internal Status PVs

### `<statusprefix>asTest`

An RPC only PV which allows testing of pvlist and ACF rules.
`$ pvcall <statusprefix>asTest pv=some:name`

Other arguments include `user="xx"`, `peer="1.1.1.1:12345`, and `roles=["yy"]`.
If omitted, the credentials of the requesting client are used.

### `<statusprefix>clients`

  A list of clients connected to the server side

### `<statusprefix>cache`

  A list of channels to which the client side is connected

### `<statusprefix>us:bypv:tx` `<statusprefix>us:bypv:rx` `<statusprefix>ds:bypv:tx` `<statusprefix>ds:bypv:rx` `<statusprefix>us:byhost:tx` `<statusprefix>us:byhost:rx` `<statusprefix>ds:byhost:tx` `<statusprefix>ds:byhost:rx`

  Each is a table showing bandwidth usage aggregated in various ways.

  'us' vs. 'ds' is upstream/server side vs. downstream/client side

  'bypv' vs. 'byhost' groups results by the PV name involved, or the peer host.
  'us:byhost:*' is grouped by upstream server (IOC).  'ds:byhost:*' is grouped
  by downstream client.

  'tx' vs. 'rx' is direction of data flow as seen by the gateway process.

## PV List file syntax

The goal is to remain compatible with the PV list file format of the cagateway.

However, at present only "EVALUATION ORDER ALLOW, DENY" is implemented,
and DENY matches always take precedence over ALLOW matches.


## ACF File syntax

The goal is to remain compatible with the ACF file parser in EPICS Base.
Some extensions are made.

At present, Get and Monitor operations are always allowed.
This is equivalent to an implied "RULE(1, READ)" in every ASG.
Explicit READ rules are treated as no-ops.

### Additional RULE permission grants

In addition to READ and WRITE, the additional permissions PUT, RPC, and UNCACHED may be granted.

The PUT and RPC permissions allow clients to perform these operations.

As a convenience, the WRITE permission is an alias which allows both PUT and RPC.

Granting the UNCACHED permission allows a client to request bypassing of
the gateway sharing and de-duplication of Monitor and Get.  A client
requests this with "record[cache=false]".  This is considered an expert
activity, and not recommended as a default.
