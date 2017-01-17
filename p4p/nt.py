
import time
from .wrapper import Type, Value

class NTScalar(object):
    Value = Value
    @staticmethod
    def buildType(valtype):
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
        ])

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
