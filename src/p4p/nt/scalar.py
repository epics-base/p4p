
import time
import sys
import numpy

from ..wrapper import Type, Value
from .common import alarm, timeStamp

if sys.version_info >= (3, 0):
    unicode = str


class ntwrappercommon(object):
    raw = timestamp = None

    def _store(self, value):
        assert isinstance(value, Value), value
        self.raw = value
        self.severity = value.get('alarm.severity', 0)
        self.status = value.get('alarm.status', 0)
        S, NS = value.get('timeStamp.secondsPastEpoch', 0), value.get('timeStamp.nanoseconds', 0)
        self.raw_stamp = S, NS
        self.timestamp = S + NS * 1e-9
        # TODO: unpack display/control
        return self

    def __str__(self):
        V = super(ntwrappercommon, self).__repr__()
        return '%s %s' % (time.ctime(self.timestamp), V)

    tostr = __str__

class ntfloat(ntwrappercommon, float):
    """
    Augmented float with additional attributes

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """


class ntint(ntwrappercommon, int):
    """
    Augmented integer with additional attributes

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """


class ntstr(ntwrappercommon, unicode):
    """
    Augmented string with additional attributes

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """


class ntnumericarray(ntwrappercommon, numpy.ndarray):
    """
    Augmented numpy.ndarray with additional attributes

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """

    @classmethod
    def build(klass, val):
        assert len(val.shape) == 1, val.shape
        # clone
        return klass(shape=val.shape, dtype=val.dtype, buffer=val.data,
                     strides=val.strides)


class ntstringarray(ntwrappercommon, list):
    """
    Augmented list of strings with additional attributes

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """

def _metaHelper(F, valtype, display=False, control=False, valueAlarm=False):
    isnumeric = valtype[-1:] not in '?su'
    if display and isnumeric:
        F.extend([
            ('display', ('S', None, [
                ('limitLow', valtype[-1:]),
                ('limitHigh', valtype[-1:]),
                ('description', 's'),
                ('format', 's'),
                ('units', 's'),
            ])),
        ])
    if control and isnumeric:
        F.extend([
            ('display', ('S', None, [
                ('limitLow', valtype[-1:]),
                ('limitHigh', valtype[-1:]),
                ('minStep', valtype[-1:]),
            ])),
        ])
    if valueAlarm and isnumeric:
        F.extend([
            ('valueAlarm', ('S', None, [
                ('active', '?'),
                ('lowAlarmLimit', valtype[-1:]),
                ('lowWarningLimit', valtype[-1:]),
                ('highWarningLimit', valtype[-1:]),
                ('highAlarmLimit', valtype[-1:]),
                ('lowAlarmSeverity', 'i'),
                ('lowWarningSeverity', 'i'),
                ('highWarningSeverity', 'i'),
                ('highAlarmSeverity', 'i'),
                ('hysteresis', 'd'),
            ])),
        ])

class NTScalar(object):

    """Describes a single scalar or array of scalar values and associated meta-data

    >>> stype = NTScalar('d') # scalar double
    >>> V = stype.wrap(4.2)
    >>> assert isinstance(V, Value)

    >>> stype = NTScalar.buildType('ad') # vector double
    >>> V = Value(stype, {'value': [4.2, 4.3]})

    The result of `wrap()` is an augmented value object combining
    `ntwrappercommon` and a python value type (`str`, `int`, `float`, `numpy.ndarray`).

    Agumented values have some additional attributes including:

    * .timestamp - The update timestamp is a float representing seconds since 1 jan 1970 UTC.
    * .raw_stamp - A tuple of (seconds, nanoseconds)
    * .severity - An integer in the range [0, 3]
    * .raw - The complete underlying :class:`~p4p.Value`
    """
    Value = Value

    @staticmethod
    def buildType(valtype, extra=[], display=False, control=False, valueAlarm=False):
        """Build a Type

        :param str valtype: A type code to be used with the 'value' field.  See :ref:`valuecodes`
        :param list extra: A list of tuples describing additional non-standard fields
        :param bool display: Include optional fields for display meta-data
        :param bool control: Include optional fields for control meta-data
        :param bool valueAlarm: Include optional fields for alarm level meta-data
        :returns: A :py:class:`Type`
        """
        isarray = valtype[:1] == 'a'
        F = [
            ('value', valtype),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
        ]
        _metaHelper(F, valtype, display=display, control=control, valueAlarm=valueAlarm)
        F.extend(extra)
        return Type(id="epics:nt/NTScalarArray:1.0" if isarray else "epics:nt/NTScalar:1.0",
                    spec=F)

    def __init__(self, valtype='d', **kws):
        self.type = self.buildType(valtype, **kws)

    def wrap(self, value, timestamp=None):
        """Pack python value into Value

        Accepts dict to explicitly initialize fields be name.
        Any other type is assigned to the 'value' field.
        """
        if isinstance(value, Value):
            return value
        elif isinstance(value, ntwrappercommon):
            return value.raw
        elif isinstance(value, dict):
            return self.Value(self.type, value)
        else:
            S, NS = divmod(float(timestamp or time.time()), 1.0)
            return self.Value(self.type, {
                'value': value,
                'timeStamp': {
                    'secondsPastEpoch': S,
                    'nanoseconds': NS * 1e9,
                },
            })

    typeMap = {
        int: ntint,
        float: ntfloat,
        unicode: ntstr,
        numpy.ndarray: ntnumericarray.build,
        list: ntstringarray,
    }

    @classmethod
    def unwrap(klass, value):
        """Unpack a Value into an augmented python type (selected from the 'value' field)
        """
        assert isinstance(value, Value), value
        V = value.value
        try:
            T = klass.typeMap[type(V)]
        except KeyError:
            raise ValueError("Can't unwrap value of type %s" % type(V))
        try:
            return T(value.value)._store(value)
        except Exception as e:
            raise ValueError("Can't construct %s around %s (%s): %s" % (T, value, type(value), e))

    def assign(self, V, py):
        """Store python value in Value
        """
        V.value = py

if sys.version_info < (3, 0):
    class ntlong(ntwrappercommon, long):
        pass

    NTScalar.typeMap[long] = ntlong
