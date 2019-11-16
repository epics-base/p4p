from __future__ import print_function

import unittest
import sys
import random
import weakref
import gc
import threading

from ..server import Server, installProvider, removeProvider, DynamicProvider, StaticProvider
from ..client.thread import Context
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

class TestServerConf(RefTestCase):
    def test_bad_iface(self):
        P = StaticProvider('x')
        with self.assertRaisesRegexp(RuntimeError, "invalid IP or non-existant hostname"):
            S = Server(providers=[P], useenv=False, conf={
                'EPICS_PVAS_INTF_ADDR_LIST':'invalid.host.name.',
                'EPICS_PVAS_BROADCAST_PORT':'0',
                'EPICS_PVAS_SERVER_PORT':'0',
            })
