# 2nd PVA gateway prototype

## Building

```sh
git clone --branch 7.0 --recursive https://github.com/epics-base/epics-base
cd epics-base/modules/pvAccess
git remote add md https://github.com/mdavidsaver/pvAccessCPP
git fetch md
git co md/gwdev
cd ../..
# now in top level of git checkout
make
```

```sh
cd ..
# now in directory containing 'epics-base'.
```

Other python environment containers, or pip install --user,
should work equally well.

```sh
virtualenv pydeps
. pydeps/bin/activate
```

```sh
git clone --branch gw https://github.com/mdavidsaver/p4p-dev p4p
cd p4p

# any of the requirements-*.txt files should work
pip install -r requirements-deb9.txt

cat <<EOF > configure/RELEASE.local
EPICS_BASE=\$(TOP)/../epics-base
EOF
make

# run tests
# also prints PYTHONPATH=...
# which needs to be used:
make nose
```

## Running

```sh
PYTHONPATH=... python -m p4p.gw -h
```

> usage: gw.py [-h] [--server SERVER] [--cip CIP] [--cport CPORT]
>              [--pvlist PVLIST] [--access ACCESS] [--prefix PREFIX]
>              [--statsdb STATSDB] [-v] [--debug]
> 
> optional arguments:
>   -h, --help         show this help message and exit
>   --server SERVER    Server interface address, with optional port (default
>                      5076)
>   --cip CIP          Space seperated client address list, with optional ports
>                      (default set by --cport)
>   --cport CPORT      Client default port
>   --pvlist PVLIST    Optional PV list file. Default allows all
>   --access ACCESS    Optional ACF file. Default allows all
>   --prefix PREFIX    Prefix for status PVs
>   --statsdb STATSDB  SQLite3 database file for stats
>   -v, --verbose
>   --debug

The argument '--cip' sets EPICS_PVA_ADDR_LIST for the client side of the
gateway.  It can be any combination of broadcast and unicast addresses.

'--server' must be a single local interface address, and optionally ':port'.

'--pvlist' and '--access' should work as with cagateway.
pvlist ALIAS lines aren't yet supported.

When '--prefix <name>' is given, which is recommended, the server side of
the gateway will present some additionaly PVs with internal status information.
Running with '-v' will print a full list of these names.  They include:

<name>asTest

  An RPC only PV which allows testing of ACF pvlist and ACF rules.
  $ pvcall <name>asTest pv=some:name

<name>clients

  A list of clients connected to the server side

<name>cache

  A list of channels to which the client side is connected

<name>us:bypv:tx
<name>us:bypv:rx
<name>ds:bypv:tx
<name>ds:bypv:rx
<name>us:byhost:tx
<name>us:byhost:rx
<name>ds:byhost:tx
<name>ds:byhost:rx

  Each is a table showing bandwidth usage aggregated in various ways.

  'us' vs. 'ds' is upstream/server side vs. downstream/client side

  'bypv' vs. 'byhost' groups results by the PV name involved, or the peer host.
  'us:byhost:*' is grouped by upstream server (IOC).  'ds:byhost:*' is grouped
  by downstream client.

  'tx' vs. 'rx' is direction of data flow as seen by the gateway process.
