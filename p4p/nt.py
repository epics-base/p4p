
import time
from .wrapper import Type, Value

class NTScalar(object):
    Value = Value
    @staticmethod
    def buildType(valtype, extra=[]):
        return Type(id="epics:nt/NTScalar:1.0",
                    spec=[
            ('value', valtype),
            ('alarm', ('s', None, [
                ('severity', 'i'),
                ('status', 'i'),
                ('message', 's'),
            ])),
            ('timeStamp', ('s', None, [
                ('secondsPastEpoch', 'l'),
                ('nanoseconds', 'i'),
                ('userTag', 'i'),
            ])),
        ]+extra)

    def __init__(self, valtype='d'):
        self.type = self.buildType(valtype)

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
    Value = Value
    @staticmethod
    def buildType(valtype, extra=[]):
        return Type(id="epics:nt/NTMultiChannel:1.0",
                    spec=[
            ('value', valtype),
            ('channelName', 'as'),
            ('descriptor', 's'),
            ('alarm', ('s', None, [
                ('severity', 'i'),
                ('status', 'i'),
                ('message', 's'),
            ])),
            ('timeStamp', ('s', None, [
                ('secondsPastEpoch', 'l'),
                ('nanoseconds', 'i'),
                ('userTag', 'i'),
            ])),
            ('severity', 'ai'),
            ('status', 'ai'),
            ('message', 'as'),
            ('secondsPastEpoch', 'al'),
            ('nanoseconds', 'ai'),
            ('userTag', 'ai'),
            ('isConnected', 'a?'),
        ]+extra)

class NTTable(object):
    Value = Value
    @staticmethod
    def buildType(columns=[], extra=[]):
        return Type(id="epics:nt/NTTable:1.0",
                    spec=[
            ('labels', 'as'),
            ('value', ('s', None, columns)),
            ('descriptor', 's'),
            ('alarm', ('s', None, [
                ('severity', 'i'),
                ('status', 'i'),
                ('message', 's'),
            ])),
            ('timeStamp', ('s', None, [
                ('secondsPastEpoch', 'l'),
                ('nanoseconds', 'i'),
                ('userTag', 'i'),
            ])),
        ]+extra)
