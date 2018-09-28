
from __future__ import print_function

import logging
_log = logging.getLogger(__name__)

import sys

from . import Server, StaticProvider
from .thread import SharedPV
from .. import nt
from .. import set_debug

defs = {
    'int': nt.NTScalar('l').wrap(0),
    'uint': nt.NTScalar('L').wrap(0),
    'real': nt.NTScalar('d').wrap(0),
    'str': nt.NTScalar('s').wrap(''),
    'aint': nt.NTScalar('al').wrap([]),
    'auint': nt.NTScalar('aL').wrap([]),
    'areal': nt.NTScalar('ad').wrap([]),
    'astr': nt.NTScalar('as').wrap([]),
    'enum': nt.NTEnum().wrap(0),
}


def getargs():
    from argparse import ArgumentParser

    P = ArgumentParser()
    P.add_argument('-d', '--debug', action='store_const', const=logging.DEBUG, default=logging.INFO)
    P.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO)
    # P.add_argument('-f', '--file', help='Persistence file')

    P.add_argument('pvs', metavar='name=def', nargs='+', help='PV definitions')

    return P.parse_args()


def main(args):
    db = {}
    provider = StaticProvider('soft')

    for pv in args.pvs:
        name, sep, type = pv.partition('=')
        if sep == '':
            print("Invalid definition, missing '=' :", pv)
            sys.exit(1)

        pv = SharedPV(initial=defs[type])

        @pv.put
        def handler(pv, op):
            pv.post(op.value())
            op.done()

        provider.add(name, pv)
        _log.info("Add %s", name)

    Server.forever(providers=[provider])

if __name__ == '__main__':
    args = getargs()
    set_debug(args.debug)
    logging.basicConfig(level=args.verbose)
    main(args)
