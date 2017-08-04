#!/usr/bin/ python

from p4p.client.thread import Context
import time

l=[]

def cb(V):
	l.append(V)

cntxt = Context('pva')

sub = cntxt.monitor('pv:SCAN', cb)
time.sleep(3)

print(l)

sub.close()
cntxt.close()