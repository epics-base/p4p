#!/usr/bin/env python3

from collections import OrderedDict
import time
import logging
import gc

from p4p import Value
from p4p.client.thread import Context

class Tracker:
    def __init__(self, pv:str, ctxt:Context, n:int, pvReq:str):
        self.prev = None
        self.nwakes = 0
        self.nupdate = 0
        self.nskip = 0
        self.S = ctxt.monitor(pv, self._update,
                              request=pvReq,
                              batch_limit=n,
                              notify_disconnect=True)

    def _update(self, u):
        self.nwakes += 1
        if isinstance(u, Value):
            u = [u]

        if isinstance(u, list):
            for v in u:
                cnt = v.value
                if not isinstance(cnt, (int, float)):
                    cnt = cnt[0]
                cnt = int(cnt)
                self.nupdate += 1
                if self.prev is not None:
                    diff = (cnt - self.prev)&0xffffffff
                    if diff!=1:
                        self.nskip += 1
                self.prev = cnt

        elif self.S:
            print(self.S.name, 'Event', u)

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-w', '--wait', metavar='sec.', type=float, default=10.0)
    P.add_argument('-H', '--ham', metavar='PV', action='append', default=[])
    P.add_argument('-S', '--spam', metavar='PV', action='append', default=[])
    P.add_argument('-P', '--pipeline', dest='pipeline', action='store_true', default=None)
    P.add_argument('-p', '--no-pipeline', dest='pipeline', action='store_false')
    P.add_argument('-Q', '--queueSize', metavar='CNT', type=int, default=10)
    return P

def main(args):
    #gc.set_debug(gc.DEBUG_SAVEALL)
    logging.basicConfig(level=logging.INFO)
    pvReq = [
        'queueSize=%d'%args.queueSize,
    ]
    if args.pipeline is not None:
        pvReq.append('pipeline='+('true' if args.pipeline else 'false'))
    pvReq = 'record[%s]'%(','.join(pvReq))
    print('pvRequest', pvReq)

    ctxt = Context(nt=False)

    trackers = OrderedDict()

    T0 = time.monotonic()
    for L in (args.ham, args.spam):
        for name in L:
            trackers[name] = Tracker(name, ctxt, args.queueSize, pvReq=pvReq)

    #gc.disable()
    #print(gc.get_stats())
    try:
        for n in range(10):
            time.sleep(args.wait/10)
            for T in trackers.values():
                print(f'{time.monotonic()-T0:.2f}: {T.S._S.stats(reset=True)}')
    except KeyboardInterrupt:
        pass
    #gc.enable()

    for T in trackers.values():
        print(T.S._S.stats())
        T.S.close()
    T1 = time.monotonic()

    dT = T1 - T0
    print('run time', dT, 'sec')

    for name, T in trackers.items():
        print(name, T.nwakes/dT, 'wakes/s', T.nupdate/dT, 'updates/s', T.nskip/dT, 'skips/s')

    #print(gc.get_stats())
    #print(gc.garbage)

if __name__=='__main__':
    main(getargs().parse_args())
