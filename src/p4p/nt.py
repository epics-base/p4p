
import logging
_log = logging.getLogger(__name__)

import time
from operator import itemgetter
from .wrapper import Type, Value

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
        
        :param valtype str: A type code to be used with the 'value' field.
        :param extra list: A list of tuples describing additional non-standard fields
        :param display bool: Include optional fields for display meta-data
        :param control bool: Include optional fields for control meta-data
        :param valueAlarm bool: Include optional fields for alarm level meta-data
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
        if isinstance(value, dict):
            return self.Value(self.type, value)
        else:
            return self.Value(self.type, {
                'value': value,
                'timeStamp': {
                    'secondsPastEpoch': time.time(),
                },
            })

class NTMultiChannel(object):
    """Describes a structure holding the equivalent of a number of NTScalar
    """
    Value = Value
    @staticmethod
    def buildType(valtype, extra=[]):
        """Build a Type
        
        :param valtype str: A type code to be used with the 'value' field.  Must be an array
        :param extra list: A list of tuples describing additional non-standard fields
        :returns: A :py:class:`Type`
        """
        assert valtype[:1]=='a', 'valtype must be an array'
        return Type(id="epics:nt/NTMultiChannel:1.0",
                    spec=[
            ('value', valtype),
            ('channelName', 'as'),
            ('descriptor', 's'),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
            ('severity', 'ai'),
            ('status', 'ai'),
            ('message', 'as'),
            ('secondsPastEpoch', 'al'),
            ('nanoseconds', 'ai'),
            ('userTag', 'ai'),
            ('isConnected', 'a?'),
        ]+extra)

class NTTable(object):
    """A generic table

    >>> table = NTTable.buildType(columns=[
        ('columnA', 'ai'),
        ('columnB', 'as'),
    ])
    """
    Value = Value
    @staticmethod
    def buildType(columns=[], extra=[]):
        """Build a table
        
        :param columns list: List of columns
        :param extra list: A list of tuples describing additional non-standard fields
        :returns: A :py:class:`Type`
        """
        return Type(id="epics:nt/NTTable:1.0",
                    spec=[
            ('labels', 'as'),
            ('value', ('s', None, columns)),
            ('descriptor', 's'),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
        ]+extra)

    def __init__(self, columns=[], extra=[]):
        self.labels = []
        C = []
        for col, type in columns:
            if type[0]=='a':
                raise ValueError("NTTable column types may not be array")
            C.append((col, 'a'+type))
            self.labels.append(col)
        self.type = self.buildType(C, extra=extra)

    def wrap(self, values):
        """Pack an iterable of dict into a Value
        
        >>> T=NTTable([('A', 'ai'), ('B', 'as')])
        >>> V = T.wrap([
            {'A':42, 'B':'one'},
            {'A':43, 'B':'two'},
        ])
        """
        cols = dict([(L, []) for L in self.labels])
        try:
            # unzip list of dict
            for V in values:
                for L in self.labels:
                    try:
                        cols[L].append(V[L])
                    except (IndexError, KeyError):
                        pass
            # allow omit empty columns
            for L in self.labels:
                V = cols[L]
                if len(V)==0:
                    del cols[L]

            try:
                return Value(self.type, {
                    'labels': self.labels,
                    'value': cols,
                })
            except:
                _log.error("Failed to encode '%s' with %s", cols, self.labels)
                raise
        except:
            if hasattr(values[0], 'keys'):
                _log.error("Columns")
            _log.exception("Failed to wrap: %s", values)
            raise
