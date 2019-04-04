import logging
import sys
import atexit

try:
    # give a chance to adjust DSO loader path
    import epicscorelibs.path
except ImportError:
    pass

from .wrapper import Value, Type
from ._p4p import (pvdVersion, pvaVersion, listRefs, Cancelled, ClientProvider as _ClientProvider)

from ._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                   logLevelInfo, logLevelWarn, logLevelError,
                   logLevelFatal, logLevelOff)

_log = logging.getLogger(__name__)

__all__ = (
    'Value',
    'Type',
)

_lvls = {
    logLevelAll, # 0
    logLevelTrace,
    logLevelDebug,
    logLevelInfo,
    logLevelWarn,
    logLevelError,
    logLevelFatal,
    logLevelOff, # 7
}

_lvlmap = {
    logging.DEBUG:logLevelDebug, # 10 -> 2
    logging.INFO:logLevelInfo,
    logging.WARN:logLevelWarn,
    logging.ERROR:logLevelError,
    logging.FATAL:logLevelFatal,
}

def set_debug(lvl):
    """Set PVA global debug print level.  This prints directly to stdout,
    bypassing eg. sys.stdout.

    :param lvl: logging.* level or logLevel*
    """
    lvl = _lvlmap.get(lvl, lvl)
    assert lvl in _lvls, lvl
    _ClientProvider.set_debug(lvl)

version = (1, 0, -80)

def cleanup():
    """P4P sequenced shutdown.  Intended to be atexit.  Idenpotent.
    """
    _log.debug("P4P atexit begins")
    # clean provider registry
    from .server import clearProviders, _cleanup_servers
    clearProviders()

    # close client contexts
    from .client.raw import _cleanup_contexts
    _cleanup_contexts()

    # stop servers
    _cleanup_servers()

    # shutdown default work queue
    from .util import _defaultWorkQueue
    _defaultWorkQueue.stop()
    _log.debug("P4P atexit completes")

atexit.register(cleanup)
