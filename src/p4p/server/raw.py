
import logging
_log = logging.getLogger(__name__)

from .._p4p import SharedPV as _SharedPV

__all__ = (
    'SharedPV',
    'Handler',
)


class ServOpWrap(object):

    def __init__(self, op, wrap, unwrap):
        self._op, self._wrap, self._unwrap = op, wrap, unwrap

    def value(self):
        V = self._op.value()
        try:
            return self._unwrap(V)
        except: # py3 will chain automatically, py2 won't
            raise ValueError("Unable to unwrap %r with %r"%(V, self._unwrap))

    def done(self, value=None, error=None):
        if value is not None:
            try:
                value = self._wrap(value)
            except:
                raise ValueError("Unable to wrap %r with %r"%(value, self._wrap))
        self._op.done(value, error)

    def __getattr__(self, key):
        return getattr(self._op, key) # dispatch to _p4p.ServerOperation


class Handler(object):
    """Skeleton of SharedPV Handler

    Use of this as a base class is optional.
    """
    def open(self, value, **kws):
        """
        Called each time an Open operation is performed on this Channel

        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).
        """

    def put(self, pv, op):
        """
        Called each time a client issues a Put
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param ServerOperation op: The operation being initiated.
        """
        op.done(error='Not supported')

    def post(self, pv, value, **kws):
        """
        Called each time a client issues a post
        operation on this Channel.

        :param SharedPV pv: The :py:class:`SharedPV` which this Handler is associated with.
        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).        
        :param dict options: A dictionary of configuration options.
        """
        pass

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

    def close(self, pv):
        """
        Called when the Channel is closed.

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
                 nt=None, wrap=None, unwrap=None,
                 options=None, **kws):
        self.nt = nt
        self._handler = handler or self._DummyHandler()
        self._whandler = self._WrapHandler(self, self._handler)

        self._wrap = wrap or (nt and nt.wrap) or (lambda x: x)
        self._unwrap = unwrap or (nt and nt.unwrap) or (lambda x: x)

        _SharedPV.__init__(self, self._whandler, options)
        if initial is not None:
            self.open(initial, nt=nt, wrap=wrap, unwrap=unwrap, **kws)

    def open(self, value, nt=None, wrap=None, unwrap=None, **kws):
        """Mark the PV as opened an provide its initial value.
        This initial value is later updated with post().

        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).

        Any clients which have begun connecting which began connecting while
        this PV was in the close'd state will complete connecting.

        Only those fields of the value which are marked as changed will be stored.
        """

        self._wrap = wrap or (nt and nt.wrap) or self._wrap
        self._unwrap = unwrap or (nt and nt.unwrap) or self._unwrap

        # Intercept all arguments that start with 'handler_open_' and remove them from
        # the arguments that go to the wrap and send them instead to the handler.open()
        post_kws = {x: kws.pop(x) for x in [y for y in kws if y.startswith("handler_open_")]}

        try:
            V = self._wrap(value, **kws)
        except: # py3 will chain automatically, py2 won't
            raise ValueError("Unable to wrap %r with %r and %r"%(value, self._wrap, kws))


        # Guard goes here because we can have handlers that don't inherit from 
        # the Handler base class
        try:
            self._handler.open(V, **post_kws)
        except AttributeError as err:
            pass

        _SharedPV.open(self, V)

    def post(self, value, **kws):
        """Provide an update to the Value of this PV.

        :param value:  A Value, or appropriate object (see nt= and wrap= of the constructor).

        Only those fields of the value which are marked as changed will be stored.

        Any keyword arguments are forwarded to the NT wrap() method (if applicable).
        Common arguments include: timestamp= , severity= , and message= .
        """
        # Intercept all arguments that start with 'handler_post_' and remove them from
        # the arguments that go to the wrap and send them instead to the handler.post()
        post_kws = {x: kws.pop(x) for x in [y for y in kws if y.startswith("handler_post_")]}

        try:
            V = self._wrap(value, **kws)
        except: # py3 will chain automatically, py2 won't
            raise ValueError("Unable to wrap %r with %r and %r"%(value, self._wrap, kws))

        # Guard goes here because we can have handlers that don't inherit from 
        # the Handler base class
        try:  
            self._handler.post(self, V, **post_kws)
        except AttributeError:
            pass

        _SharedPV.post(self, V)

    def close(self, destroy=False, sync=False, timeout=None):
        """Close PV, disconnecting any clients.

        :param bool destroy: Indicate "permanent" closure.  Current clients will not see subsequent open().
        :param bool sync: When block until any pending onLastDisconnect() is delivered (timeout applies).
        :param float timeout: Applies only when sync=True.  None for no timeout, otherwise a non-negative floating point value.

        close() with destory=True or sync=True will not prevent clients from re-connecting.
        New clients may prevent sync=True from succeeding.
        Prevent reconnection by __first__ stopping the Server, removing with :py:meth:`StaticProvider.remove()`,
        or preventing a :py:class:`DynamicProvider` from making new channels to this SharedPV.
        """
        try:  
            self._handler.close(self)
        except AttributeError:
            pass

        _SharedPV.close(self)

    def current(self):
        V = _SharedPV.current(self)
        try:
            return self._unwrap(V)
        except: # py3 will chain automatically, py2 won't
            raise ValueError("Unable to unwrap %r with %r"%(V, self._unwrap))

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

        def open(self, value, **kws):
            _log.debug('OPEN %s %s', self._pv, value)
            try:
                self._pv._exec(None, self._real.open, value, **kws)
            except AttributeError:
                pass

        def onFirstConnect(self):
            _log.debug('ONFIRSTCONNECT %s', self._pv)
            self._pv._exec(None, self._pv._onFirstConnect, None)
            try:  # user handler may omit onFirstConnect()
                M = self._real.onFirstConnect
            except AttributeError:
                return
            self._pv._exec(None, M, self._pv)

        def onLastDisconnect(self):
            _log.debug('ONLASTDISCONNECT %s', self._pv)
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
                self._pv._exec(op, self._real.put, self._pv, ServOpWrap(op, self._pv._wrap, self._pv._unwrap))
            except AttributeError:
                op.done(error="Put not supported")

        def rpc(self, op):
            _log.debug('RPC %s %s', self._pv, op)
            try:
                self._pv._exec(op, self._real.rpc, self._pv, op)
            except AttributeError:
                op.done(error="RPC not supported")

        def post(self, value, **kws):
            _log.debug('POST %s %s', self._pv, value)
            try:
                self._pv._exec(None, self._real.rpc, self._pv, value, **kws)
            except AttributeError:
                pass

        def close(self, pv):
            _log.debug('CLOSE %s', self._pv)
            try:
                self._pv._exec(None, self._real.close, self._pv)
            except AttributeError:
                pass

    @property
    def on_open(self):
        def decorate(fn):
            self._handler.open = fn
            return fn
        return decorate        

    @property
    def on_first_connect(self):
        def decorate(fn):
            self._handler.onFirstConnect = fn
            return fn
        return decorate

    @property
    def on_last_disconnect(self):
        def decorate(fn):
            self._handler.onLastDisconnect = fn
            return fn
        return decorate

    @property
    def on_put(self):
        def decorate(fn):
            self._handler.put = fn
            return fn
        return decorate

    @property
    def on_rpc(self):
        def decorate(fn):
            self._handler.rpc = fn
            return fn
        return decorate
  
    @property
    def on_post(self):
        def decorate(fn):
            self._handler.post = fn
            return fn
        return decorate        

    @property
    def on_close(self):
        def decorate(fn):
            self._handler.close = fn
            return fn
        return decorate   

    # Aliases for decorators to maintain consistent new style
    # Required because post is already used and on_post seemed the best
    # alternative.
    put = on_put
    rpc = on_rpc
    onFirstConnect = on_first_connect
    onLastDisconnect = on_last_disconnect

    def __repr__(self):
        if self.isOpen():
            return '%s(value=%s)' % (self.__class__.__name__, repr(self.current()))
        else:
            return "%s(<closed>)" % (self.__class__.__name__,)
    __str__ = __repr__
