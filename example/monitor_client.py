#!/usr/bin/env python


import sys, time, logging

from p4p.client.thread import Context

logging.basicConfig(level=logging.DEBUG)

def cb(value):
    print("update", value)

print("Create Context")
with Context('pva') as ctxt:
    print("Subscribe to", sys.argv[1])
    S = ctxt.monitor(sys.argv[1], cb)

    print("Waiting")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass

    print("Close subscription")
    S.close()
print("Done")
