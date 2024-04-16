#!/usr/bin/env python

# cf.
# example/monitor_client.py


import sys, time, logging

from p4p.client.thread import Context

logging.basicConfig(level=logging.INFO)

def cb(value):
    if not value.raw.changed('value'):
        print("Meta update")
        for fld in value.raw.asSet():
            print(" ",fld,value.raw[fld])


print("Create Context")
with Context('pva') as ctxt:
    print("Subscribe to", sys.argv[1])
    S = ctxt.monitor(sys.argv[1], cb)

    print("Waiting")
    try:
        time.sleep(50)
    except KeyboardInterrupt:
        pass

    print("Close subscription")
    S.close()
print("Done")
