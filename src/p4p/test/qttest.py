import logging
import unittest
from functools import wraps
from math import ceil

from qtpy.QtCore import QObject, QCoreApplication

from ..client.Qt import exceptionGuard, Disconnected, TimeoutError, Context

from ..server import Server, StaticProvider
from ..server.thread import SharedPV, _defaultWorkQueue
from ..nt import NTScalar
from .utils import RefTestCase

_log = logging.getLogger(__name__)

class waitSignal(QObject):
    def __init__(self, sig=None):
        QObject.__init__(self)
        self._T = None
        self._args = None
        self._sig = sig

    def __enter__(self):
        if self._sig is not None:
            self._sig.connect(self.slot)
        return self

    def __exit__(self,A,B,C):
        self.close()
        if self._sig is not None:
            self._sig.disconnect(self.slot)

    def close(self):
        if self._T is not None:
            self.killTimer(self._T)
            self._T = None

    @exceptionGuard
    def timerEvent(self, evt):
        _log.error('Test timeout')
        QCoreApplication.instance().quit()
        self.close()

    def slot(self, *args):
        _log.debug('signalled %s', args)
        self._args = args
        QCoreApplication.instance().quit()
        self.close()

    def wait(self, timeout=5.0):
        assert self._T is None, self._T
        self._T = self.startTimer(ceil(timeout*1000))
        try:
            self._args = None
            QCoreApplication.instance().exec_()
            if self._args is not None:
                return self._args

            else:
                raise TimeoutError()
        finally:
            self.close()

    def waitValue(self, **kws):
        E, = self.wait(**kws)
        if isinstance(E, Exception):
            raise E
        else:
            return E

class TestGPM(RefTestCase):
    def setUp(self):
        super(TestGPM, self).setUp()

        self._timedout = False
        self.app = QCoreApplication([])

        self.pv = SharedPV(nt=NTScalar('i'), initial=42)
        @self.pv.put
        def mailbox(pv, op):
            _log.debug('onPut: %s', op.value())
            pv.post(op.value())
            op.done()
        self.server = Server(providers=[{'pvname':self.pv}], isolate=True)

        self.ctxt = Context('pva', conf=self.server.conf(), useenv=False)

    def tearDown(self):
        self.ctxt.close()

        self.server.stop()
        _defaultWorkQueue.sync()
        del self.ctxt
        del self.server
        del self.pv
        del self.app
        super(TestGPM, self).tearDown()

    def test_pm(self):
        with waitSignal() as sub, waitSignal() as put:
            S = self.ctxt.monitor('pvname', sub.slot)
            E, = sub.wait()
            self.assertIsInstance(E, Disconnected)

            self.assertEqual(sub.waitValue(), 42)

            self.ctxt.put('pvname', 51, slot=put.slot)
            put.waitValue()

            self.assertEqual(sub.waitValue(), 51)
