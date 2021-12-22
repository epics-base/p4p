
import time
from datetime import datetime
from ..wrapper import Type

if hasattr(datetime, 'timestamp'):
    _dt2posix = datetime.timestamp

else: # py 2.x
    from calendar import timegm as _timegm
    def _dt2posix(dt):
        return _timegm(dt.timetuple()) * dt.microsecond*1e-6

# common sub-structs
timeStamp = Type(id='time_t', spec=[
    ('secondsPastEpoch', 'l'),
    ('nanoseconds', 'i'),
    ('userTag', 'i'),
])
alarm = Type(id='alarm_t', spec=[
    ('severity', 'i'),
    ('status', 'i'),
    ('message', 's'),
])

class NTBase(object):
    """Helper for NT* (un)wrapper classes
    """
    @staticmethod
    def buildType():
        """Return a Type
        """
        raise NotImplementedError("NT classes must provide buildType()")

    def __init__(self, *args, **kws):
        self.type = self.buildType(*args, **kws)

    def wrap(self, pyval, **kws):
        """Called during eg. SharedPV.post() to build a Value
        """
        raise NotImplementedError("NT classes must provide wrap()")

    def assign(self, V, pyval):
        """Called during eg. Context.put() to update the Value V from the provided python object
        """
        return self.wrap(self.buildType(), pyval)

    def unwrap(self, V):
        """Translate a Value into an appropriate python object.  (the reverse of wrap())
        """
        raise NotImplementedError("NT classes must provide unwrap()")

    @staticmethod
    def _annotate(value, timestamp=None, severity=None, message=None):
        """Update and return provided 'value'.

        Updates timeStamp.secondsPastEpoch, timeStamp.nanoseconds with
        the provided timestamp or the current system time.
        """
        if severity is not None:
            value['alarm.severity'] = severity

        if message is not None:
            value['alarm.message'] = message

        if timestamp is not None:
            # timestamp may be: datetime, seconds as float or int, or tuple of (sec, ns)
            if isinstance(timestamp, datetime):
                timestamp = _dt2posix(timestamp)

            if isinstance(timestamp, (int, float)):
                sec, ns = divmod(timestamp, 1.0)
                timestamp = (int(sec), int(ns*1e9))

            # at this point timestamp must be a tuple of (sec, ns)
            value['timeStamp'] = {'secondsPastEpoch':timestamp[0], 'nanoseconds':timestamp[1]}
        return value
