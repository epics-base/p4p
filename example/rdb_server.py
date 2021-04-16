#!/usr/bin/env python
"""
Example of syncing data from an RDB.

In this case, with bundled sqlite, with a schema meant
to be representative of an accelerator lattice description.

Includes both a table PV of multiple elements, and
individual PVs for element attributes.

1. <prefix>TBL

May be monitored to sync a copy of the lattice DB.
An RPC may be used for filtered queries.

2. <prefix><element>(type|S|L|foo)

Access to attributes of individual elements.

A full list of PV names is printed on startup.
"""

from __future__ import print_function

import logging
import sqlite3
import time

from p4p.nt import NTTable, NTScalar
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV

_log = logging.getLogger(__name__)

tableType = NTTable(columns=[
    ('name', 's'),
    ('type', 's'),
    ('S', 'd'),
    ('L', 'd'),
])

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('db', help='sqlite database file (will be created if missing')
    P.add_argument('prefix', help='PV name prefix')
    P.add_argument('-d', '--debug', action='store_const', const=logging.DEBUG, default=logging.INFO)
    return P

def main(args):
    logging.basicConfig(level=args.debug)

    elements = {}
    with sqlite3.connect(args.db) as C:
        ver, = C.execute('PRAGMA user_version').fetchone()
        _log.info('schema version %s', ver)
        if ver==0:
            # dummy lattice schema
            _log.info('Initialize %s', args.db)
            C.executescript('''
CREATE TABLE elements (
    name STRING NOT NULL UNIQUE,
    type STRING NOT NULL,
    S REAL NOT NULL,
    L REAL NOT NULL DEFAULT 1,
    foo REAL NOT NULL DEFAULT 0
);
INSERT INTO elements(name,type, S, L) VALUES ('gun', 'source', 0, 0);
INSERT INTO elements(name,type, S) VALUES ('drift1', 'drift', 0);
INSERT INTO elements(name,type, S) VALUES ('Q1', 'quad', 1);
INSERT INTO elements(name,type, S) VALUES ('drift2', 'drift', 2);
INSERT INTO elements(name,type, S) VALUES ('Q2', 'quad', 3);
INSERT INTO elements(name,type, S) VALUES ('drift3', 'drift', 4);
INSERT INTO elements(name,type, S) VALUES ('Q3', 'quad', 5);
PRAGMA user_version = 1;
''')

        elif ver!=1:
            raise RuntimeError('unsupported user_version %s', ver)


        prov = StaticProvider('rdb')
        # publish complete table
        table = SharedPV(nt=tableType, initial=[])

        # allow RPC to filter table
        @table.rpc
        def query(pv, op):
            params = op.value().query # cf. NTURI
            Q, A = 'SELECT name, type, S, L FROM elements WHERE 0=0', []
            for col in ('name', 'type', 'S', 'L'):
                if col in params:
                    Q += ' AND %s=?'%col
                    A.append(params[col])

            with sqlite3.connect(args.db) as C:
                C.row_factory = sqlite3.Row
                op.done(tableType.wrap(C.execute(Q, A)))

        prov.add(args.prefix+'TBL', table)

        # also publish elements (excepting drifts) individually
        for name, in C.execute("SELECT name FROM elements WHERE type!='drift'"):
            pvs = {}
            for ptype, initial, param in (('s', '', 'type'), ('s', '', 'S'), ('d', 0, 'L'), ('d', 0, 'foo')):
                pv = SharedPV(nt=NTScalar(ptype), initial=initial)
                prov.add('%s%s:%s'%(args.prefix, name, param), pv)
                pvs[param] = pv
            elements[name] = pvs

    # list PVs being served
    print('Serving')
    for pv in prov.keys():
        print(' ', pv)

    with Server(providers=[prov]):
        while True:
            # periodically re-sync
            # assumes elements not added/removed (simplification)
            with sqlite3.connect(args.db) as C:
                C.row_factory = sqlite3.Row

                all = list(C.execute('SELECT name, type, S, L FROM elements ORDER BY S ASC'))

                table.post(all)

                for name, type, S, L in all:
                    if name in elements:
                        elements[name]['type'].post(type)
                        elements[name]['S'].post(S)
                        elements[name]['L'].post(L)

            time.sleep(2.0)

if __name__=='__main__':
    main(getargs().parse_args())
