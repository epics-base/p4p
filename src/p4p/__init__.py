import logging
from .wrapper import Value, Type
from ._p4p import (pvdVersion, pvaVersion, listRefs, Cancelled, ClientProvider as _ClientProvider)

from ._p4p import (logLevelAll, logLevelTrace, logLevelDebug,
                   logLevelInfo, logLevelWarn, logLevelError,
                   logLevelFatal, logLevelOff)

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

version = (1, 1, 0)
