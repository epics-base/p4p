#!/usr/bin/ python

import logging
logging.basicConfig()

import time, numpy
from p4p.client.thread import Context

Get_Times = []
Get_Values = []
P = 'pv:0'

cntxt = Context('pva')

for x in range(0, 10):
	t0 = time.time()
	Get_Values.append(cntxt.get(P))
	t1 = time.time()
	Get_Times.append(t1-t0)

print('''	min = {0}
	max = {1}
	mean = {2}
	stdev = {3}''').format(min(Get_Times), max(Get_Times), numpy.mean(Get_Times), numpy.std(Get_Times))

cntxt.close()