import _cython_3_2_5
import p4p._p4p
from _typeshed import Incomplete
from typing import Any

Server_report: _cython_3_2_5.cython_function_or_method
__reduce_cython__: _cython_3_2_5.cython_function_or_method
__setstate_cython__: _cython_3_2_5.cython_function_or_method
__test__: dict
addOdometer: _cython_3_2_5.cython_function_or_method

class Channel(InfoBase):
    expired: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def access(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CreateOp(InfoBase):
    """Handle for in-progress Channel creation request
    """
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def create(self, *args, **kwargs):
        """Create a Channel with a given upstream (server-side) name

                :param bytes name: Upstream name to use.  This is what the GW Client will search for.
                :returns: A `Channel`
        """
    def __reduce__(self): ...

class InfoBase:
    account: Incomplete
    peer: Incomplete
    roles: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class Provider(p4p._p4p.Source):
    BanHost: Incomplete
    BanHostPV: Incomplete
    BanPV: Incomplete
    Claim: Incomplete
    Ignore: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def cachePeek(self, *args, **kwargs):
        """Returns PV names in channel cache

                :returns: a set of strings
        """
    def clearBan(self, *args, **kwargs):
        """Clear the negative results cache
        """
    def forceBan(self, *args, **kwargs):
        """Preemptively Add an entry to the negative result cache.
                Either host or usname must be not None

                :param bytes host: None or a host name
                :param bytes usname: None or a upstream (Server side) PV name
        """
    def ignoreByGUID(self, *args, **kwargs): ...
    def report(self, *args, **kwargs):
        """Run Client/Upstream bandwidth usage report

                :returns: List of tuple
                :rtype: [(usname, opTx, opRx, peer, trTx, trRx)]
        """
    def stats(self, *args, **kwargs):
        """Return statistics of various internal caches

                :rtype: dict
        """
    def sweep(self, *args, **kwargs):
        """Call periodically to remove unused `Channel` from channel cache.
        """
    def testChannel(self, usname) -> Any:
        """testChannel(usname)
                Add the upstream name to the channel cache and begin trying to connect.
                Returns Claim if the channel is connected, and Ignore if it is not.

                :param bytes usname: Upstream (Server side) PV name
                :returns: Claim or Ignore
        """
    def use_count(self, *args, **kwargs): ...
    def __reduce__(self): ...
