import logging
import warnings

import unittest

from ..asLib import Engine
from ..asLib.pvlist import PVList

class TestPVList(unittest.TestCase):
    def test_slac(self):
        pvlist = """
EVALUATION ORDER ALLOW, DENY
# comment
.* ALLOW 
OTRS:DMP1:695:Image:.*     DENY
PATT:SYS0:1:MPSBURSTCTRL.* ALLOW CANWRITE
PATT:SYS0:1:MPSBURSTCTRL.* DENY FROM 1.2.3.4 
BEAM.* DENY FROM 1.2.3.4   
BEAM.* ALLOW
BEAM.* ALLOW RWINSTRMCC 1


"""

        pvl = PVList(pvlist)

        self.assertEqual(pvl.compute(b'BEAM:stuff', '127.0.0.1'), ('BEAM:stuff', 'RWINSTRMCC', 1))

        self.assertEqual(pvl.compute(b'OTHER:stuff', '127.0.0.1'), ('OTHER:stuff', 'DEFAULT', 0))

        self.assertEqual(pvl.compute(b'OTRS:DMP1:695:Image:X', '127.0.0.1'), (None, None, None))

        self.assertEqual(pvl.compute(b'PATT:SYS0:1:MPSBURSTCTRLX', '127.0.0.1'), ('PATT:SYS0:1:MPSBURSTCTRLX', 'CANWRITE', 0))
        self.assertEqual(pvl.compute(b'PATT:SYS0:1:MPSBURSTCTRLX', '1.2.3.4'), (None, None, None))

class DummyEngine(Engine):
    @staticmethod
    def _gethostbyname(host):
        return {
            'localhost':'127.0.0.1',
            'lcls-daemon3':'1.2.3.4',
            'pscag1':'1.2.3.44',
        }[host]

class TestACL(unittest.TestCase):
    class DummyChannel(object):
        def __init__(self):
            self.perm = None
        def access(self, **kws):
            self.perm = kws

    def test_default(self):
        eng = DummyEngine()

        ch = self.DummyChannel()
        eng.create(ch, 'DEFAULT', 'someone', 'somewhere', 0)
        self.assertDictEqual(ch.perm, {'put':True, 'rpc':True, 'uncached':True})

        ch = self.DummyChannel()
        eng.create(ch, 'othergrp', 'someone', 'somewhere', 0)
        self.assertDictEqual(ch.perm, {'put':True, 'rpc':True, 'uncached':True})

    def test_roles(self):
        eng = DummyEngine("""
UAG(SPECIAL) {
    root,
    group:admin
}
ASG(DEFAULT)
{
        RULE(1,READ)
        RULE(1,WRITE) {
            UAG(SPECIAL)
        }
}
""")

        for args, perm in [(('DEFAULT', 'someone', 'somewhere', 0),          {'put':False,'rpc':False, 'uncached':False}),
                           (('DEFAULT', 'root', '1.2.3.4', 0),               {'put':True, 'rpc':False, 'uncached':False}),
                           (('DEFAULT', 'someone', '1.2.3.4', 0, ['admin']), {'put':True, 'rpc':False, 'uncached':False}),
                           ]:
            try:
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
HAG(GWSTATS)
{
        lcls-daemon3
}
HAG(PHOTON)
{
        pscag1
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
""")

        for args, perm in [(('DEFAULT', 'someone', 'somewhere', 0),  {'put':False, 'rpc':False, 'uncached':False}),
                           (('DEFAULT', 'root', '1.2.3.4', 0),       {'put':False, 'rpc':False, 'uncached':False}),
                           (('CANWRITE', 'someone', 'somewhere', 0), {'put':True , 'rpc':False, 'uncached':False}),
                           (('CANWRITE', 'root', '1.2.3.44', 0),     {'put':True , 'rpc':False, 'uncached':False}),
                           (('AMOWRITE', 'someone', 'somewhere', 0), {'put':False, 'rpc':False, 'uncached':False}),
                           (('AMOWRITE', 'someone', '1.2.3.44', 0),  {'put':False, 'rpc':False, 'uncached':False}),
                           (('AMOWRITE', 'root', 'somewhere', 0),    {'put':False, 'rpc':False, 'uncached':False}),
                           (('AMOWRITE', 'root', '1.2.3.44', 0),     {'put':True , 'rpc':False, 'uncached':False}),
                           ]:
            try:
                ch = self.DummyChannel()
                eng.create(ch, *args)
                self.assertDictEqual(ch.perm, perm)
            except AssertionError as e:
                raise AssertionError('%s -> %s : %s'%(args, perm ,e))
