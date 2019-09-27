
from __future__ import print_function

import warnings
import os
import sys
import json
import logging
import shutil
from collections import OrderedDict

from . import Server, StaticProvider
from .thread import SharedPV
from .. import nt
from .. import set_debug

_log = logging.getLogger(__name__)

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
    P.add_argument('-f', '--file', help='Persistence file')

    P.add_argument('pvs', metavar='name=type', nargs='+', help='PV definitions.  types: %s'%(', '.join(defs.keys())))

    return P.parse_args()

def buildMailbox(*args, **kws):
    pv = SharedPV(*args, **kws)

    @pv.put
    def handler(pv, op):
        pv.post(op.value())
        op.done()

    return pv

def main(args):
    db = OrderedDict()
    provider = StaticProvider('soft')

    for pv in args.pvs:
        name, sep, type = pv.partition('=')
        if sep == '':
            print("Invalid definition, missing '=' :", pv)
            sys.exit(1)

        pv = buildMailbox(initial=defs[type])

        provider.add(name, pv)
        db[name] = (type, pv)
        _log.info("Add %s", name)

    if args.file:
        # pre-create to ensure write permission
        OF = open(args.file+'.tmp', 'w')

    if args.file and os.path.exists(args.file):
        with open(args.file, 'r') as F:
            persist = json.load(F)

        if persist['version']!=1:
            warnings.warn('Unknown persist version %s.  Attempting to load'%persist['version'])
        persist = persist['pvs']

        for name, type, iv in persist:
            if name not in db:
                db[name] = (type, buildMailbox(initial=defs[type]))

            _type, pv = db[name]
            pv.post(pv.current().type()(iv))

    try:
        Server.forever(providers=[provider])
    finally:
        persist = []
        for name, (type, pv) in db.items():
            persist.append((name, type, pv.current().todict(None, OrderedDict)))

        if args.file:
            OF.write(json.dumps(OrderedDict([('version',1), ('pvs',persist)]), indent=2))
            OF.flush()
            OF.close()
            shutil.move(args.file+'.tmp', args.file)

if __name__ == '__main__':
    args = getargs()
    set_debug(args.debug)
    logging.basicConfig(level=args.verbose)
    main(args)
