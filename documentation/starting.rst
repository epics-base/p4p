.. _starting:

Quick Start
===========

For testing and evaluation, it is recommended to install from pypi.org into a (disposable) virtualenv. ::

    python -m virtualenv p4ptest
    . p4ptest/bin/activate
    python -m pip install -U pip
    python -m pip install p4p
    python -m nose p4p   # Optional: runs automatic tests

With this complete, open three terminal instances.
One to run a PVA client, and two more to run PVA clients.
In the first run a PVA server.  Feel free to replace 'my:pv:name'
with an arbitrary name string everywhere it occurs. ::

    $ . p4ptest/bin/activate
    $ python -m p4p.server.cli my:pv:name=int
    ...
    INFO:p4p.server:Running server

In a second terminal run the following. If successful, the last line will end with a zero value. ::

    $ . p4ptest/bin/activate
    $ python -m p4p.client.cli monitor my:pv:name
    ...
    my:pv:name Mon Jul  9 19:24:01 2018 0L

And finally, in a third terminal run the following.  If successful, the second terminal should show the new value. ::

    $ . p4ptest/bin/activate
    $ python -m p4p.client.cli put my:pv:name=5
    my:pv:name=5

Alternately, using the `clientapi`. ::

    from p4p.client.thread import Context
    ctxt = Context('pva')
    print(ctxt.get('my:pv:name'))
    ctxt.put('my:pv:name', 6)
    print(ctxt.get('my:pv:name'))

Troubleshooting network issues
------------------------------

If the preceeding didn't work as described, there is likely a networking problem.
Restart both server and client with an added argument '-d'.
Look for lines like: ::

    ...
    2018-09-27T17:00:44.164 Broadcast address #0: 10.65.15.255:5076. (not unicast)
    2018-09-27T17:00:44.164 Broadcast address #1: 192.168.210.255:5076. (not unicast)
    ...

This indicates the two network interfaces were discovered on this host.
Localhost is not included by default.
If no interfaces are found, then check the system network configuration.
If interfaces are found, ensure that no firewalls are applied.

By default, both client and server must share at least one discovered
local broadcast address.

Non-default configuration involves the use of the $EPICS_PVA_ADDR_LIST
environment variable, or configuration options specifically passed
to conf= `p4p.client.thread.Context()`.  This is considered beyond the
scope of a "quick start".
