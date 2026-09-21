import _cython_3_2_5
from _typeshed import Incomplete
from p4p.wrapper import Type as Type, Value as Value
from typing import Any, ClassVar, Iterable, overload

__pyx_capi__: dict
__reduce_cython__: _cython_3_2_5.cython_function_or_method
__setstate_cython__: _cython_3_2_5.cython_function_or_method
__test__: dict
listRefs: _cython_3_2_5.cython_function_or_method
logLevelAll: int
logLevelDebug: int
logLevelError: int
logLevelFatal: int
logLevelInfo: int
logLevelOff: int
logLevelTrace: int
logLevelWarn: int
logger_level_set: _cython_3_2_5.cython_function_or_method
version: _cython_3_2_5.cython_function_or_method
version_str: _cython_3_2_5.cython_function_or_method

class Cancelled(RuntimeError):
    """Cancelled from client end."""
    def __init__(self, *args, **kwargs) -> None: ...

class ClientMonitor:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    handler: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def close(self, *args, **kwargs): ...
    def pop(self, *args, **kwargs): ...
    def __reduce__(self): ...

class ClientOperation:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    name: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def close(self, *args, **kwargs): ...
    def __reduce__(self): ...

class ClientProvider:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def close(self, *args, **kwargs): ...
    def conf(self, *args, **kwargs): ...
    def disconnect(self, *args, **kwargs): ...
    def hurryUp(self, *args, **kwargs): ...
    @staticmethod
    def makeRequest(*args, **kwargs): ...
    def __reduce__(self): ...

class Disconnected(RuntimeError):
    """Channel becomes disconected."""
    def __init__(self, *args, **kwargs) -> None: ...

class DynamicProvider:
    handler: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...

class Finished(Disconnected):
    """Special case of Disconnected when a Subscription has received all updates it will ever receive."""
    def __init__(self, *args, **kwargs) -> None: ...

class RemoteError(RuntimeError):
    """Thrown with an error message which has been sent by a server to its remote client"""

class Server:
    guid: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def conf(self, *args, **kwargs): ...
    def interrupt(self, *args, **kwargs): ...
    def run(self, *args, **kwargs): ...
    def start(self, *args, **kwargs): ...
    def stop(self, *args, **kwargs): ...
    def tostr(self, *args, **kwargs): ...
    def __reduce__(self): ...

class ServerOperation:
    """An in-progress Put or RPC operation from a client.
    """
    handler: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def account(self) -> str:
        """account() -> str
                Client identity
        """
    @overload
    def done(self, value=..., error=...) -> Any:
        '''done(value=None, error=None)

                Signal completion of the operation. ::

                  # successful completion without result (Put or RPC)
                  done()
                  # successful completion with result (RPC only)
                  done(Value)
                  # unsuccessful completion (Put or RPC)
                  done(error="msg")
        '''
    @overload
    def done(self) -> Any:
        '''done(value=None, error=None)

                Signal completion of the operation. ::

                  # successful completion without result (Put or RPC)
                  done()
                  # successful completion with result (RPC only)
                  done(Value)
                  # unsuccessful completion (Put or RPC)
                  done(error="msg")
        '''
    @overload
    def done(self, Value) -> Any:
        '''done(value=None, error=None)

                Signal completion of the operation. ::

                  # successful completion without result (Put or RPC)
                  done()
                  # successful completion with result (RPC only)
                  done(Value)
                  # unsuccessful completion (Put or RPC)
                  done(error="msg")
        '''
    @overload
    def done(self, error=...) -> Any:
        '''done(value=None, error=None)

                Signal completion of the operation. ::

                  # successful completion without result (Put or RPC)
                  done()
                  # successful completion with result (RPC only)
                  done(Value)
                  # unsuccessful completion (Put or RPC)
                  done(error="msg")
        '''
    def info(self, *args, **kwargs): ...
    def name(self) -> str:
        """name() -> str
                The PV name used by the client
        """
    def onCancel(self, *args, **kwargs):
        """onCancel(callable|None)

                Set callable which will be invoked if the remote operation is
                cancelled by the client, or if client connection is lost.
        """
    def peer(self) -> str:
        """peer() -> str
                Client address
        """
    def pvRequest(self) -> Value:
        """pvRequest() -> Value
                Access the request Value provided by the client, which may ignored, or used to modify handling.
        """
    def roles(self, *args, **kwargs):
        """roles() -> {str}
                Client group memberships
        """
    def value(self) -> Value:
        """value() -> Value
                For an RPC operation, the argument Value provided
        """
    def warn(self, *args, **kwargs): ...
    def __reduce__(self): ...

class SharedArray:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class SharedPV:
    handler: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def close(self, *args, **kwargs): ...
    def current(self, *args, **kwargs): ...
    def isOpen(self, *args, **kwargs): ...
    def open(self, *args, **kwargs): ...
    def post(self, *args, **kwargs): ...
    def __reduce__(self): ...

class Source:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class StaticProvider:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def add(self, *args, **kwargs): ...
    def close(self, *args, **kwargs): ...
    def keys(self, *args, **kwargs): ...
    def remove(self, *args, **kwargs): ...
    def __reduce__(self): ...

class _Type:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def aspy(self, str=...) -> list:
        """aspy(str=None) -> list
                Return a Type specification list equivalent to the one passed to the constructor.
        """
    def getID(self, *args, **kwargs):
        """getId() -> str
                Return Type id= string
        """
    def has(self, str) -> bool:
        """has(str) -> bool
                Does this Type include the named member field?
        """
    def keys(self) -> Iterable[str]:
        """keys() -> Iterable[str]
                Return child field names
        """
    def tostr(self, limit: int = ...) -> str:
        """tostr(limit : int = 0) -> str

                Return a string representation, optionally truncated to a length limit

                :param int limit: If greater than zero, formatting is terminated at ``limit`` charactors.
        """
    def __getitem__(self, index):
        """Return self[key]."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class _Value:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def changed(self, *args, **kwargs): ...
    def changedSet(self, *args, **kwargs): ...
    def get(self, key: str, default=...) -> Value | Any:
        """get(key : str, default=None) -> Value | Any
                dict-like access to sub-field

                :param str key: Sub-field name
                :param default: returned if sub-field doesn't exist
        """
    def getID(self) -> str:
        """getID() -> str
                Return Type id= string
        """
    def has(self, name: str) -> bool:
        """has(name : str) -> bool
                Test for sub-field existance

                :param str name: Sub-field name
        """
    def items(self, key: str = ...) -> Iterable[Value | Any]:
        """items(key : str = None) -> Iterable[Value | Any]

                :param str key: Sub-field name
        """
    def mark(self, field=..., val=...) -> Any:
        """mark(field=None, val=True)
                Mark (or unmark) the this field, or the named sub-field.

                :param str field: Sub-field name
                :param bool val: To mark, or unmark
        """
    def select(self, name: str, selector: str) -> Any:
        """select(name : str, selector : str)
                Explicitly select Union member

                :param str name: Sub-field name
        """
    def todict(self, name=..., wrapper=...) -> Mapping[str, Value]:
        """todict(name=None, wrapper=None) -> Mapping[str, Value]

                Return this Value (or the named sub-field) translated into a dict

                :param str name: Sub-field name, or None
                :param callable wrapper: Passed an iterable of name,value tuples.  By default ``dict``  eg. could be OrderedDict
        """
    def tolist(self, name=...) -> list[tuple[str, Value]]:
        """tolist(name=None) -> List[Tuple[str, Value]]
                Return this Value (or the named sub-field) translated into a list of tuples
        """
    def tostr(self, limit: int = ...) -> str:
        """tostr(limit : int = 0) -> str

                Return a string representation, optionally truncated to a length limit

                :param int limit: If greater than zero, formatting is terminated at ``limit`` charactors.
        """
    def type(self, fld: str = ...) -> Type:
        """type(fld : str = None) -> Type
                Return the Type of this Value, or the named sub-field.

                :param str fld: Sub-field name, or None
        """
    def unmark(self, *args, **kwargs):
        """Unmark Value and all sub-fields.
        """
    def __delattr__(self, name):
        """Implement delattr(self, name)."""
    def __delitem__(self, other) -> None:
        """Delete self[key]."""
    def __getattr__(self, key: str) -> Value | Any:
        """__getattr__(key : str) -> Value | Any

                :param str key: Sub-field name
        """
    def __getitem__(self, index):
        """items(key : str) -> Value | Any

                :param str key: Sub-field name
        """
    def __iter__(self):
        """Implement iter(self)."""
    def __reduce__(self): ...
    def __setattr__(self, key: str, value) -> Any:
        """__setattr__(key : str, value)

                :param str key: Sub-field name
                :param value: value to assign
        """
    def __setitem__(self, key: str, value) -> Any:
        """__setitem__(key : str, value)

                :param str key: Sub-field name
                :param value: value to assign
        """
