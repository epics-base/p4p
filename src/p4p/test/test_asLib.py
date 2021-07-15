import logging
import warnings
import weakref
import collections
import gc

import unittest

from ..asLib import Engine
from ..asLib.yacc import parse as parse_acf
from ..asLib.pvlist import PVList, _sub_add

from .. import nt

_log = logging.getLogger(__name__)

class TestPVList(unittest.TestCase):
    def test_slac(self):
        pvlist = r"""
EVALUATION ORDER ALLOW, DENY
# comment
.* ALLOW 
OTRS:DMP1:695:Image:.*     DENY
PATT:SYS0:1:MPSBURSTCTRL.* ALLOW CANWRITE
PATT:SYS0:1:MPSBURSTCTRL.* DENY FROM 1.2.3.4 
PATT:SYS0:1:MPSBURSTCTRL.* DENY From 4.3.2.1
X(.*) ALIAS Y\1 CANWRITE
BEAM.* DENY FROM 1.2.3.4   
BEAM.* ALLOW
BEAM.* ALLOW RWINSTRMCC 1
BEAM:L1:.* allow

THIS ALIAS THAT

a:([^:]*):b:([^:]*) ALIAS A:\2:B:\1

"""

        pvl = PVList(pvlist)

        self.assertEqual(pvl.compute(b'BEAM:stuff', '127.0.0.1'), ('BEAM:stuff', 'RWINSTRMCC', 1))
        self.assertEqual(pvl.compute(b'BEAM:L1:Energy', '127.0.0.1'), ('BEAM:L1:Energy', 'DEFAULT', 0))

        self.assertEqual(pvl.compute(b'OTHER:stuff', '127.0.0.1'), ('OTHER:stuff', 'DEFAULT', 0))

        self.assertEqual(pvl.compute(b'OTRS:DMP1:695:Image:X', '127.0.0.1'), (None, None, None))

        self.assertEqual(pvl.compute(b'PATT:SYS0:1:MPSBURSTCTRLX', '127.0.0.1'), ('PATT:SYS0:1:MPSBURSTCTRLX', 'CANWRITE', 0))
        self.assertEqual(pvl.compute(b'PATT:SYS0:1:MPSBURSTCTRLX', '1.2.3.4'), (None, None, None))
        self.assertEqual(pvl.compute(b'PATT:SYS0:1:MPSBURSTCTRLX', '4.3.2.1'), (None, None, None))

        self.assertEqual(pvl.compute(b'Xsomething', '127.0.0.1'), ('Ysomething', 'CANWRITE', 0))
        self.assertEqual(pvl.compute(b'a:one:b:two', '127.0.0.1'), ('A:two:B:one', 'DEFAULT', 0))

class LocalSubscription(object):
    def __init__(self, cb, code='d'):
        self.cb = cb
        self.nt = nt.NTScalar(code)
    def close(self):
        self.cb = None
    def post(self, V):
        if V is not None:
            V = self.nt.unwrap(self.nt.wrap(V))
        cb = self.cb
        if cb is not None:
            cb(V)

class LocalContext(object):
    def __init__(self, provider):
        assert provider=='pva', provider
        self._sub = collections.defaultdict(weakref.WeakSet)

    def monitor(self, pv, cb, notify_disconnect=False):
        assert notify_disconnect is True
        S = LocalSubscription(cb)
        S.post(None)
        self._sub[pv].add(S)
        return S

    def post(self, pv, V):
        for S in set(self._sub[pv]):
            S.post(V)

class DummyEngine(Engine):
    @staticmethod
    def _gethostbyname(host):
        return {
            'localhost':'127.0.0.1',
            'lcls-daemon3':'1.2.3.4',
            'hxr-control':'1.2.3.44',
            'hxr-console':'1.2.3.45',
            'mcrhost':'1.2.3.10',
            'remotehost':'1.2.3.20',
        }.get(host)

class TestACF(unittest.TestCase):
    def test_parse(self):
        inp='''
UAG(SPECIAL) {
    root,
    "role/admin"
}
HAG(GWSTATS)
{
        lcls-daemon3,
        other.host,
        "strange",
        "invalid.host.name."
}
ASG(SIMPLE)
{
        RULE(1,READ)
}
ASG(NOTSIMPLE) {
  INPA("ACC-CT{}Prmt:Remote-Sel")
        RULE(1,READ)
        RULE(1,WRITE,TRAPWRITE) {
                UAG(PHOTON)
                HAG(PHOTON, OTHER)
    CALC("A!=0")
                
        }
}
'''

        ast = parse_acf(inp)

        self.assertEqual(ast, [
            ('UAG', 'SPECIAL', ['root', 'role/admin']),
            ('HAG', 'GWSTATS', ['lcls-daemon3', 'other.host', 'strange', 'invalid.host.name.']),
            ('ASG', 'SIMPLE', [
                ('RULE', 1, 'READ', False, None)
            ]),
            ('ASG', 'NOTSIMPLE', [
                ('INP', 'A', 'ACC-CT{}Prmt:Remote-Sel'),
                ('RULE', 1, 'READ', False, None),
                ('RULE', 1, 'WRITE', True, [
                    ('UAG', ['PHOTON']),
                    ('HAG', ['PHOTON', 'OTHER']),
                    ('CALC', 'A!=0'),
                ]),
            ]),
        ])

class TestACL(unittest.TestCase):
    class DummyChannel(object):
        def __init__(self):
            self.perm = None
            self.name = 'dummy'
        def access(self, **kws):
            self.perm = kws

    def tearDown(self):
        # cleanup lingering refs. to Type arising through LocalSubscription
        # which would cause false positive failures in subsequent cases
        gc.collect()
        super(TestACL, self).tearDown()

    def test_default(self):
        eng = DummyEngine()
        self.assertIsNone(eng._ctxt)

        ch = self.DummyChannel()
        eng.create(ch, 'DEFAULT', 'someone', 'somewhere', 0)
        self.assertDictEqual(ch.perm, {'put':True, 'rpc':True, 'uncached':True, 'audit': False})

        ch = self.DummyChannel()
        eng.create(ch, 'othergrp', 'someone', 'somewhere', 0)
        self.assertDictEqual(ch.perm, {'put':True, 'rpc':True, 'uncached':True, 'audit': False})

    def test_roles(self):
        eng = DummyEngine("""
UAG(SPECIAL) {
    root,
    "role/admin"
}
ASG(DEFAULT)
{
        RULE(1,READ)
        RULE(1,WRITE) {
            UAG(SPECIAL)
        }
}
""")

        for args, perm in [(('DEFAULT', 'someone', 'somewhere', 0),          {'put':False,'rpc':False, 'uncached':False, 'audit': False}),
                           (('DEFAULT', 'root', '1.2.3.4', 0),               {'put':True, 'rpc':True, 'uncached':False, 'audit': False}),
                           (('DEFAULT', 'someone', '1.2.3.4', 0, ['admin']), {'put':True, 'rpc':True, 'uncached':False, 'audit': False}),
                           ]:
            try:
                _log.debug('With: %s expect: %s', args, perm)
                ch = self.DummyChannel()
                eng.create(ch, *args)
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s -> %s : %s'%(args, perm ,e))

    def test_slac(self):
        eng = DummyEngine("""
UAG(PHOTON)
{
        root
}
UAG(tst-opr)
{
        tstioc, tstopr
}
HAG(GWSTATS)
{
        lcls-daemon3
}
HAG(PHOTON)
{
        hxr-control, hxr-console,
        "invalid.host.name."
}
ASG(DEFAULT)
{
        RULE(1,READ)
}
ASG(CANWRITE)
{
        RULE(1,READ)  
        RULE(1,WRITE,TRAPWRITE)
}
ASG(AMOWRITE)
{
        RULE(1,READ)
        RULE(1,WRITE,TRAPWRITE)
                {
                UAG(PHOTON)
                HAG(PHOTON)
                }
}
ASG(TSTWRITE)
{
        RULE(1,READ)
        RULE(1,WRITE,TRAPWRITE)
                {
                UAG( PHOTON , tst-opr )
                HAG(PHOTON,GWSTATS)
                }
}
""")

        for args, perm in [(('DEFAULT', 'someone', 'somewhere', 0),  {'put':False, 'rpc':False, 'uncached':False, 'audit': False}),
                           (('DEFAULT', 'root', '1.2.3.4', 0),       {'put':False, 'rpc':False, 'uncached':False, 'audit': False}),
                           (('CANWRITE', 'someone', 'somewhere', 0), {'put':True , 'rpc':True,  'uncached':False, 'audit': True}),
                           (('CANWRITE', 'root', '1.2.3.44', 0),     {'put':True , 'rpc':True,  'uncached':False, 'audit': True}),
                           (('AMOWRITE', 'someone', 'somewhere', 0), {'put':False, 'rpc':False, 'uncached':False, 'audit': False}),
                           (('AMOWRITE', 'someone', '1.2.3.44', 0),  {'put':False, 'rpc':False, 'uncached':False, 'audit': False}),
                           (('AMOWRITE', 'root', 'somewhere', 0),    {'put':False, 'rpc':False, 'uncached':False, 'audit': False}),
                           (('AMOWRITE', 'root', '1.2.3.44', 0),     {'put':True , 'rpc':True,  'uncached':False, 'audit': True}),
                           ]:
            try:
                ch = self.DummyChannel()
                eng.create(ch, *args)
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s -> %s : %s'%(args, perm ,e))

    def test_bnl(self):
        eng = DummyEngine("""
HAG(mcr) {mcrhost}
UAG(softioc) {softioc, rtems}
HAG(remote) {remotehost}
ASG(OPERATOR) {
  INPA("ACC-CT{}Prmt:Remote-Sel")

  RULE(1, READ)

  RULE(1, WRITE, TRAPWRITE) {
    HAG(mcr)
  }

# Allow inter-IOC and physics app writes
  RULE(1, WRITE, TRAPWRITE) {
    UAG(softioc)
  }

# Conditionally allow remote consoles
  RULE(1, WRITE, TRAPWRITE) {
    HAG(remote)
    CALC("A!=0")
  }
}
""", ctxt=LocalContext('pva'))
        self.assertIsNotNone(eng._ctxt)

        db = []
        for args in [('OPERATOR', 'joe', '1.2.3.10', 0),     # always allowed MCR
                     ('OPERATOR', 'softioc', '1.2.3.99', 0), # always allowed IOC
                     ('OPERATOR', 'joe', '1.2.3.99', 0),     # never allowed
                     ('OPERATOR', 'joe', '1.2.3.20', 0),     # conditional
                     ]:
            ch = self.DummyChannel()
            eng.create(ch, *args)
            db.append(ch)

        for ch, perm in zip(db, [{'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':False , 'rpc':False, 'uncached':False, 'audit': False},
                                 {'put':False , 'rpc':False, 'uncached':False, 'audit': False},
                                 ]):
            try:
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s : %s'%(perm ,e))

        eng._ctxt.post('ACC-CT{}Prmt:Remote-Sel', 1.0)

        for ch, perm in zip(db, [{'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':False , 'rpc':False, 'uncached':False, 'audit': False},
                                 {'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 ]):
            try:
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s : %s'%(perm ,e))

        eng._ctxt.post('ACC-CT{}Prmt:Remote-Sel', 0.0)

        for ch, perm in zip(db, [{'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':True , 'rpc':True, 'uncached':False, 'audit': True},
                                 {'put':False , 'rpc':False, 'uncached':False, 'audit': False},
                                 {'put':False , 'rpc':False, 'uncached':False, 'audit': False},
                                 ]):
            try:
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s : %s'%(perm ,e))

class TestRE(unittest.TestCase):
    def test_offset(self):
        self.assertEqual(_sub_add(r'test', 1, 5), r'test')
        self.assertEqual(_sub_add(r'tes\\t', 1, 5), r'tes\\t')
        self.assertEqual(_sub_add(r'tes\1t', 1, 5), r'tes\6t')
        self.assertEqual(_sub_add(r'tes\1t\2', 2, 5), r'tes\6t\7')
