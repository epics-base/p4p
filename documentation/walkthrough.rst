Walkthrough
=======================

Using a Debian machine.

Create EPICS IOC files. 

**test0.cmd** ::

	dbLoadRecords("test0.db", "INST=pv")
	iocInit()

**test0.db** ::

	record(ai,"$(INST):0"){
		field(VAL,0)
		field(UDF,1)
	}

	record(calc,"$(INST):SCAN"){
		field(VAL,0)
		field(UDF,1)
		field(CALC,"(A>1000)?0:A+1")
		field(INPA,"$(INST):0")
		field(SCAN,"1 second")
		field(DESC, "$(INST):0")
	}

To start the IOC Server, first make the softIocPVA command available, by including the path in our command. ::

	$ /path/to/p4p/pvaSrv/bin/$HOST_ARCH/softIocPVA test0.cmd


Export the p4p module to the PYTHONPATH. ::

	$ export PYTHONPATH=$PYTHONPATH:/path/to/p4p/$PYTHON_VERSION/$HOST_ARCH