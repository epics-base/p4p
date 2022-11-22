import logging
import sys
import atexit

try:
    # give a chance to adjust DSO loader path
    import epicscorelibs
except ImportError:
    pass
else:
    import epicscorelibs.path
    import pvxslibs.path

from .wrapper import Value, Type
from ._p4p import (version as pvxsVersion, listRefs, logger_level_set as _logger_level_set)

from ._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                   logLevelInfo, logLevelWarn, logLevelError,
                   logLevelFatal, logLevelOff)
from .version import version

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

def pvdVersion():
    # ~equivalent
    return (8, 0, 0, 0)

def pvaVersion():
    # ~equivalent
    return (7, 0, 0, 0)

def set_debug(lvl):
    """Set PVA global debug print level.  This prints directly to stdout,
    bypassing eg. sys.stdout.

    :param lvl: logging.* level or logLevel*
    """
    lvl = _lvlmap.get(lvl, lvl)
    assert lvl in _lvls, lvl
    _logger_level_set("p4p.*", lvl)

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
