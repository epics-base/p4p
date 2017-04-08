
import logging
_log = logging.getLogger(__name__)

import atexit, sys
from weakref import WeakSet

try:
    from Queue import Queue, Full, Empty
except ImportError:
    from queue import Queue, Full, Empty

from ._p4p import (Context as _Context,
                   Channel as _Channel)
from ._p4p import logLevelDebug

__all__ = (
    'Context',
)

class Channel(_Channel):
    name = property(_Channel.getName)

class Context(_Context):
    Channel = Channel

    def __init__(self, *args, **kws):
        _Context.__init__(self, *args, **kws)
        _all_contexts.add(self)
        # we keep strong refs here to shadow the Channel refs
        # from the underlying Provider, which we can't get at.
        self._channels = set()

    def close(self):
        for ch in self._channels:
            ch.close()
        self._channels = None
        _Context.close(self)

    def channel(self, name):
        if self._channels is None:
            raise ValueError("Context closed")
        ch = _Context.channel(self, name)
        self._channels.add(ch)
        return ch

_all_contexts = WeakSet()

def _cleanup_contexts():
    contexts = list(_all_contexts)
    for ctxt in contexts:
        ctxt.close()

atexit.register(_cleanup_contexts)


def getargs():
    from argparse import ArgumentParser
    A = ArgumentParser()
    A.add_argument('pv', nargs='*', help="PV names")
    A.add_argument('-P','--provider',default='pva')
    A.add_argument('-d','--debug', action='store_true')
    A.add_argument('-t','--timeout', type=float, default=5.0)
    return A.parse_args()

def main(args):
    if args.debug:
        Context.set_debug(logLevelDebug)
    ctxt = Context(args.provider)
    chans = []

    Q = Queue(maxsize=len(args.pv))
    for pv in args.pv:
        chan = ctxt.channel(pv)
        def got(V, pv=pv):
            try:
                Q.put_nowait((pv, V))
            except:
                _log.exception("Error w/ result")
        op = chan.get(got)
        chans.append(chan) # keep Channel alive

    err = 0
    for n in range(len(args.pv)):
        pv, V = Q.get(timeout=args.timeout)
        if isinstance(V, Exception):
            print pv, V
            err = 1
        else:
            print pv, V.tolist()

    del chans

    sys.exit(err)

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    main(args)
