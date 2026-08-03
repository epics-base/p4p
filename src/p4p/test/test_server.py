from __future__ import print_function

import unittest
import sys
import random
import weakref
import gc
import threading

from ..server import Server, installProvider, removeProvider, DynamicProvider, StaticProvider
from ..server.thread import SharedPV
from ..client.thread import Context
from ..nt import NTScalar
from .utils import RefTestCase


def checkweak(O):
    o = O()
    if o is not None:
        print('Live object', id(o), type(o), sys.getrefcount(o), gc.get_referrers(o))
    return o


class TestDummyProvider(RefTestCase):
    # will fail if anything is done to it.

    class Dummy(object):
        name = "foo"

    def test_install(self):
        "Install and remove provider"
        D = self.Dummy()
        d = weakref.ref(D)
        P = DynamicProvider(D.name, D)
        p = weakref.ref(P)
        installProvider("foo", P)
        del D
        del P
        removeProvider("foo")
        gc.collect()
        self.assertIsNone(d())
        self.assertIsNone(p())

    def test_server(self):
        D = self.Dummy()
        d = weakref.ref(D)
        P = DynamicProvider(D.name, D)
        p = weakref.ref(P)
        installProvider("foo", P)
        try:
            with Server(providers=["foo"]) as S:
                s = weakref.ref(S)
        finally:
            removeProvider("foo")
        del D
        del P
        del S
        gc.collect()
        self.assertIsNone(checkweak(d))
        self.assertIsNone(p())
        self.assertIsNone(checkweak(s))

    def test_server_direct(self):
        D = self.Dummy()
        d = weakref.ref(D)
        P = DynamicProvider(D.name, D)
        p = weakref.ref(P)
        with Server(providers=[P]) as S:
            s = weakref.ref(S)
        del D
        del P
        del S
        gc.collect()
        self.assertIsNone(checkweak(d))
        self.assertIsNone(p())
        self.assertIsNone(checkweak(s))

    def test_client(self):
        D = self.Dummy()
        d = weakref.ref(D)
        P = DynamicProvider(D.name, D)
        p = weakref.ref(P)
        installProvider("foo", P)
        del D
        gc.collect()
        self.assertIsNotNone(d())

        try:
            with Context('server:foo'):
                removeProvider("foo")
                # Our DynamicProvider will not longer be found by new Contexts
                # however, it remains active so long as 'P' is active

            del P
            gc.collect()
            self.assertIsNone(d())
            self.assertIsNone(p())
        finally:
            removeProvider("foo")

class TestMultipleProviders(RefTestCase):
    class Single(DynamicProvider):
        def __init__(self, pv):
            super(TestMultipleProviders.Single, self).__init__(pv, self)
            self._name = pv
            self._pv = SharedPV(nt=NTScalar('s'), initial=pv)
        def testChannel(self, name):
            return name==self._name
        def makeChannel(self, name, peer):
            if name==self._name:
                return self._pv

    def test_multiple(self):
        with Server([self.Single('one'), self.Single('two')], isolate=True) as S:
            with Context('pva', conf=S.conf(), useenv=False) as C:
                self.assertEqual('one', C.get('one'))
                self.assertEqual('two', C.get('two'))

class TestServerConf(RefTestCase):
    def test_bad_iface(self):
        P = StaticProvider('x')
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            S = Server(providers=[P], useenv=False, conf={
                'EPICS_PVAS_INTF_ADDR_LIST':'invalid.host.name.',
                'EPICS_PVAS_BROADCAST_PORT':'0',
                'EPICS_PVAS_SERVER_PORT':'0',
            })
