
import logging
import warnings
_log = logging.getLogger(__name__)

from functools import partial

from threading import Thread

from .._p4p import SharedPV as _SharedPV
from ..client.raw import LazyRepr

__all__ = (
    'SharedPV',
        'Handler',
)


class ServOpWrap(object):

    def __init__(self, op, unwrap):
        self._op, self._unwrap = op, unwrap

    def value(self):
        return self._unwrap(self._op.value())

    def __getattr__(self, key):
        return getattr(self._op, key)


class Handler(object):

    """Skeleton of SharedPV Handler

    Use of this as a base class is optional.
    """

    def put(self, pv, op):
        """
        Called each time a client issues a Put
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param ServerOperation op: The operation being initiated.
        """
        op.done(error='Not supported')

    def rpc(self, pv, op):
        """
        Called each time a client issues a Remote Procedure Call
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param ServerOperation op: The operation being initiated.
        """
        op.done(error='Not supported')

    def onFirstConnect(self, pv):
        """
        Called when the first Client channel is created.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        """
        pass

    def onLastDisconnect(self, pv):
        """
        Called when the last Client channel is closed.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        """
        pass


class SharedPV(_SharedPV):

    """Shared state Process Variable.  Callback based implementation.

    .. note:: if initial=None, the PV is initially **closed** and
              must be :py:meth:`open()`'d before any access is possible.

    :param handler: A object which will receive callbacks when eg. a Put operation is requested.
                    May be omitted if the decorator syntax is used.
    :param Value initial: An initial Value for this PV.  If omitted, :py:meth:`open()`s must be called before client access is possible.
    :param nt: An object with methods wrap() and unwrap().  eg :py:class:`p4p.nt.NTScalar`.
    :param callable wrap: As an alternative to providing 'nt=', A callable to transform Values passed to open() and post().
    :param callable unwrap: As an alternative to providing 'nt=', A callable to transform Values returned Operations in Put/RPC handlers.
    :param dict options: A dictionary of configuration options.

    Creating a PV in the open state, with no handler for Put or RPC (attempts will error). ::

        from p4p.nt import NTScalar
        pv = SharedPV(nt=NTScalar('d'), value=0.0)
        # ... later
        pv.post(1.0)

    The full form of a handler object is: ::

        class MyHandler:
            def put(self, pv, op):
                pass
            def rpc(self, pv, op):
                pass
            def onFirstConnect(self): # may be omitted
                pass
            def onLastDisconnect(self): # may be omitted
                pass
    pv = SharedPV(MyHandler())

    Alternatively, decorators may be used. ::

        pv = SharedPV()
        @pv.put
        def onPut(pv, op):
            pass

    The nt= or wrap= and unwrap= arguments can be used as a convience to allow
    the open(), post(), and associated Operation.value() to be automatically
    transform to/from :py:class:`Value` and more convienent Python types.
    See :ref:`unwrap`
    """

    def __init__(self, handler=None, initial=None,
                 nt=None, wrap=None, unwrap=None, **kws):
        self.nt = nt
        self._handler = handler or self._DummyHandler()
        self._whandler = self._WrapHandler(self, self._handler)

        self._wrap = wrap or (nt and nt.wrap) or (lambda x: x)
        self._unwrap = unwrap or (nt and nt.unwrap) or (lambda x: x)

        _SharedPV.__init__(self, self._whandler, **kws)
        if initial is not None:
            self.open(initial, nt=nt, wrap=wrap, unwrap=unwrap)

    def open(self, value, nt=None, wrap=None, unwrap=None):
        """Mark the PV as opened an provide its initial value.
        This initial value is later updated with post().

        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).

        Any clients which have begun connecting which began connecting while
        this PV was in the close'd state will complete connecting.

        Only those fields of the value which are marked as changed will be stored.
        """

        self._wrap = wrap or (nt and nt.wrap) or self._wrap
        self._unwrap = unwrap or (nt and nt.unwrap) or self._unwrap

        _SharedPV.open(self, self._wrap(value))

    def post(self, value):
        """Provide an update to the Value of this PV.

        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).

        Only those fields of the value which are marked as changed will be stored.
        """
        _SharedPV.post(self, self._wrap(value))

    def current(self):
        return self._unwrap(_SharedPV.current(self))

    def _exec(self, op, M, *args):  # sub-classes will replace this
        try:
            M(*args)
        except Exception as e:
            if op is not None:
                op.done(error=str(e))
            _log.exception("Unexpected")

    def _onFirstConnect(self, _junk):
        pass # see sub-classes.  run before user onFirstConnect()

    def _onLastDisconnect(self, _junk):
        pass # see sub-classes.  run after user onLastDisconnect()

    class _DummyHandler(object):
        pass

    class _WrapHandler(object):

        "Wrapper around user Handler which logs exceptions"

        def __init__(self, pv, real):
            self._pv = pv  # this creates a reference cycle, which should be collectable since SharedPV supports GC
            self._real = real

        def onFirstConnect(self):
            self._pv._exec(None, self._pv._onFirstConnect, None)
            try:  # user handler may omit onFirstConnect()
                M = self._real.onFirstConnect
            except AttributeError:
                return
            self._pv._exec(None, M, self._pv)

        def onLastDisconnect(self):
            try:
                M = self._real.onLastDisconnect
            except AttributeError:
                pass
            else:
                self._pv._exec(None, M, self._pv)
            self._pv._exec(None, self._pv._onLastDisconnect, None)

        def put(self, op):
            _log.debug('PUT %s %s', self._pv, op)
            try:
                self._pv._exec(op, self._real.put, self._pv, ServOpWrap(op, self._pv._unwrap))
            except AttributeError:
                op.done(error="Put not supported")

        def rpc(self, op):
            _log.debug('RPC %s %s', self._pv, op)
            try:
                self._pv._exec(op, self._real.rpc, self._pv, op)
            except AttributeError:
                op.done(error="RPC not supported")

    @property
    def onFirstConnect(self):
        def decorate(fn):
            self._handler.onFirstConnect = fn
        return decorate

    @property
    def onLastDisconnect(self):
        def decorate(fn):
            self._handler.onLastDisconnect = fn
        return decorate

    @property
    def put(self):
        def decorate(fn):
            self._handler.put = fn
        return decorate

    @property
    def rpc(self):
        def decorate(fn):
            self._handler.rpc = fn
        return decorate

    def __repr__(self):
        if self.isOpen():
            return '%s(value=%s)' % (self.__class__.__name__, repr(self.current()))
        else:
            return "%s(<closed>)" % (self.__class__.__name__,)
    __str__ = __repr__
