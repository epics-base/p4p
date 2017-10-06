.. _starting:

Getting Started
===============

The following assumes a PVAccess server providing a PV named 'test:array' which is an NTScalarArray.
See below for an example of how to set this up.


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

Demo IOC Setup
--------------

If the pva2pva module is built (see :ref:`builddeps`) then create a demo database file: ::

    cat <<EOF > p4p-demo.db
    record(ao, "test:scalar") {}
    record(waveform, "test:array") {
        field(FTVL, "DOUBLE")
        field(NELM, "100") # max element count
    }
    EOF
    ./pva2pva/bin/linux-x86_64/softIocPVA -d p4p-demo.db
