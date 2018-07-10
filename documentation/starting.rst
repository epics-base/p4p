.. _starting:

Getting Started
===============

For testing and evaluation, it is recommended to install from pypi.org into a (disposable) virtualenv. ::

    $ virtualenv p4ptest
    $ . p4ptest/bin/activate
    $ python -m pip install -U pip
    $ python -m pip install p4p
    $ python -m nose p4p   # Optional: runs automatic tests

With this complete, open three terminal instances.
In the first run a PVA server.  Feel free to replace 'my:pv:name'
with an arbitrary name string everywhere it occurs. ::

    $ python -m p4p.server.cli my:pv:name=int
    ...
    INFO:p4p.server:Running server

In a second terminal run the following. If successful, the last line will end with a zero value. ::

    $ python -m p4p.client.cli monitor my:pv:name
    ...
    my:pv:name Mon Jul  9 19:24:01 2018 0L

And finally, in a third terminal run the following.  If successful, the second terminal should show the new value. ::

    $ pthon -m p4p.client.cli put my:pv:name=5
    my:pv:name=5

Troubleshooting network issues
------------------------------

If the preceeding didn't work as described, there is likely a networking problem.

TODO: Troubleshooting procedure

Client API Usage
----------------

First start a server in another terminal. ::

    $ python -m p4p.server.cli test:array=areal

Now start python interactive shell.  Either run 'python', or if available 'ipython'.

Import and create a client :py:class:`p4p.client.thread.Context` for the PVAccess protocol. ::

    from p4p.client.thread import Context
    ctxt = Context('pva')

Now issue a :py:meth:`p4p.client.thread.Context.get` network operation for a PV named 'test:array'. ::

    val = ctxt.get('test:array')
    print val

Due to :ref:`unwrap` (which can be disabled) the type of 'val' is a :py:class:`p4p.nt.scalar.ntnumericarray`.
Which can be treated as a normal numpy array with additional attributes to access the timestamp and alarm information. ::

    print val[:5]
    print val.timestamp
    print val.severity

The underlying :py:class:`p4p.Value` can be accessed with the 'raw' attribute.
For types which are not automatically unwrap, a :py:class:`p4p.Value` is returned. ::

    print val.raw.value
    print val.raw.alarm.severity

To change a value with a :py:meth:`p4p.client.thread.Context.put` operation. ::

    ctxt.put('test:array', [1,2,3])
    val = ctxt.get('test:array')
    print val

To start and stop a subscription with :py:meth:`p4p.client.thread.Context.monitor`
which returns a :py:class:`p4p.client.thread.Subscription`. ::

   def show(val):
      print val
   S = ctxt.monitor('test:array', show)
   # current value is printed when monitor subscription is created
   ctxt.put('test:array', [4,5,6])
   # subscription update with new value is printed
   S.close() # end subscription
