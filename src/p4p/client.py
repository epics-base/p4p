
import atexit
from weakref import WeakSet

from ._p4p import Context as _Context
from ._p4p import logLevelDebug

__all__ = (
    'Context',
)

_all_contexts = WeakSet()

def _cleanup_contexts():
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()

atexit.register(_cleanup_contexts)

class Context(_Context):
    def __init__(self, *args, **kws):
        _Context.__init__(self, *args, **kws)
        _all_contexts.add(self)



def getargs():
    from argparse import ArgumentParser
    A = ArgumentParser()
    A.add_argument('pv', nargs='*', help="PV names")
    A.add_argument('-P','--provider',default='pva')
    A.add_argument('-d','--debug', action='store_true')
    return A.parse_args()

def main(args):
    from threading import Event
    if args.debug:
        Context.set_debug(logLevelDebug)
    ctxt = Context(args.provider)
    chans = []
    for pv in args.pv:
        chan = ctxt.channel(pv)
        E = Event()
        R = [None]
        def got(V):
            R[0] = V
            E.set()
        op = chan.get(got)
        chans.append((pv, chan, E, R))

    for pv, chan, E, R in chans:
        E.wait()
        if isinstance(R[0], Exception):
            print pv, R[0]
        else:
            print pv, R[0].tolist()

if __name__=='__main__':
    main(getargs())
