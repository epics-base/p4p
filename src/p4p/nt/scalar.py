
import time
import numpy

from ..wrapper import Type, Value
from .common import alarm, timeStamp

_doc = """
Has additional attributes

* .severity
* .status
* .timestamp
* .raw_stamp
"""

class ntwrappercommon(object):
    raw = timestamp = None
    def _store(self, value):
        assert isinstance(value, Value), value
        self.raw = value
        self.severity = value.get('alarm.severity', 0)
        self.status = value.get('alarm.status', 0)
        S, NS = value.get('timeStamp.secondsPastEpoch', 0), value.get('timeStamp.nanoseconds', 0)
        self.raw_stamp = S, NS
        self.timestamp = S+NS*1e-9
        # TODO: unpack display/control
        return self
    def __str__(self):
        V = super(ntwrappercommon, self).__repr__()
        return '%s %s'%(time.ctime(self.timestamp), V)

class ntfloat(ntwrappercommon,float):
    "Augmented float"+_doc
    pass

class ntint(ntwrappercommon,int):
    "Augmented int"+_doc
    pass

class ntstr(ntwrappercommon,str):
    "Augmented str"+_doc
    pass

class ntnumericarray(ntwrappercommon,numpy.ndarray):
    "Augmented numpy.ndarray"+_doc

    @classmethod
    def build(klass, val):
        assert len(val.shape)==1, val.shape
        # clone
        return klass(shape=val.shape, dtype=val.dtype, buffer=val.data,
                     strides=val.strides)

class ntstringarray(ntwrappercommon,list):
    "Augmented list (of strings)"+_doc
    pass

class NTScalar(object):
    """Describes a single scalar or array of scalar values and associated meta-data
    
    >>> stype = NTScalar.buildType('d') # scalar double
    >>> V = Value(stype, {'value': 4.2})

    >>> stype = NTScalar.buildType('ad') # vector double
    >>> V = Value(stype, {'value': [4.2, 4.3]})
    """
    Value = Value

    @staticmethod
    def buildType(valtype, extra=[], display=False, control=False, valueAlarm=False):
        """Build a Type
        
        :param str valtype: A type code to be used with the 'value' field.
        :param list extra: A list of tuples describing additional non-standard fields
        :param bool display: Include optional fields for display meta-data
        :param bool control: Include optional fields for control meta-data
        :param bool valueAlarm: Include optional fields for alarm level meta-data
        :returns: A :py:class:`Type`
        """
        isarray = valtype[:1]=='a'
        F = [
            ('value', valtype),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
        ]
        if display and valtype not in '?su':
            F.extend([
                ('display', ('s', None, [
                    ('limitLow', valtype[-1:]),
                    ('limitHigh', valtype[-1:]),
                    ('description', 's'),
                    ('format', 's'),
                    ('units', 's'),
                ])),
            ])
        if control and valtype not in '?su':
            F.extend([
                ('display', ('s', None, [
                    ('limitLow', valtype[-1:]),
                    ('limitHigh', valtype[-1:]),
                    ('minStep', valtype[-1:]),
                ])),
            ])
        if valueAlarm and valtype not in '?su':
            F.extend([
                ('valueAlarm', ('s', None, [
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
        F.extend(extra)
        return Type(id="epics:nt/NTScalarArray:1.0" if isarray else "epics:nt/NTScalar:1.0",
                    spec=F)

    def __init__(self, valtype='d', **kws):
        self.type = self.buildType(valtype, **kws)

    def wrap(self, value):
        """Pack python value into Value
        
        Accepts dict to explicitly initialize fields be name.
        Any other type is assigned to the 'value' field.
        """
        if isinstance(value, dict):
            return self.Value(self.type, value)
        else:
            return self.Value(self.type, {
                'value': value,
                'timeStamp': {
                    'secondsPastEpoch': time.time(),
                },
            })

    typeMap = {
        int: ntint,
        float: ntfloat,
        str: ntstr,
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
            raise ValueError("Can't unwrap value of type %s"%type(V))
        try:
            return T(value.value)._store(value)
        except Exception as e:
            raise ValueError("Can't construct %s around %s (%s): %s"%(T, value, type(value), e))
