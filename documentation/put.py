#!/usr/bin/ python

import logging
logging.basicConfig()

import time, numpy, sys
from p4p.client.thread import Context

Put_Times = []
Put_Values = []
Get_Values = []
P = 'pv:0'

cntxt = Context('pva')

for x in range(0, 10):
	t0 = time.time()
	cntxt.put(P, x)
	t1 = time.time()
	Put_Values.append(x)
	Get_Values.append(cntxt.get(P))
	Put_Times.append(t1-t0)

if Put_Values != Get_Values:
	print('Put and Get values mismatch')
	sys.exit(1)

else:
	print('''	min = {0}
	max = {1}
	mean = {2}
	stdev = {3}''').format(min(Put_Times), max(Put_Times), numpy.mean(Put_Times), numpy.std(Put_Times))

cntxt.close()