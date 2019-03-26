
from __future__ import absolute_import

import sys
import time

from ..wrapper import Type, Value
from .common import alarm, timeStamp
from .scalar import _metaHelper, ntwrappercommon

if sys.version_info >= (3, 0):
    unicode = str

class ntenum(ntwrappercommon, int):
    """

    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """

    def _store(self, value):
        ntwrappercommon._store(self, value)
        self.choice = None

        return self

    def __str__(self):
        return self.choice or int.__str__(self)

    def __repr__(self):
        return '%s(%d, %s)'%(self.__class__.__name__, int(self), self.choice)

    # TODO: compare with str

class NTEnum(object):
    """Describes a string selected from among a list of possible choices.  Stored internally as an integer
    """
    @staticmethod
    def buildType(extra=[], display=False, control=False, valueAlarm=False):
        F = [
            ('value', ('S', 'enum_t', [
                ('index', 'i'),
                ('choices', 'as'),
            ])),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
        ]
        # TODO: different metadata options
        _metaHelper(F, 'i', display=display, control=control, valueAlarm=valueAlarm)
        F.extend(extra)
        return Type(id="epics:nt/NTEnum:1.0", spec=F)

    def __init__(self, **kws):
        self.type = self.buildType(**kws)
        # cached choices list.
        # picked up during unwrap
        self._choices = []

    def wrap(self, value, timestamp=None):
        """Pack python value into Value
        """
        V = self.type()
        S, NS = divmod(float(timestamp or time.time()), 1.0)
        V.timeStamp = {
            'secondsPastEpoch': S,
            'nanoseconds': NS * 1e9,
        }
        if isinstance(value, dict):
            # assume dict of index and choices list
            V.value = value
            self._choices = V['value.choices']
        else:
            # index or string
            self.assign(V, value)
        return V

    def unwrap(self, value):
        """Unpack a Value into an augmented python type (selected from the 'value' field)
        """
        if value.changed('value.choices'):
            self._choices = value['value.choices']

        idx = value['value.index']
        ret = ntenum(idx)._store(value)
        try:
            ret.choice = self._choices[idx]
        except IndexError:
            pass # leave it as None
        return ret

    def assign(self, V, py):
        """Store python value in Value
        """
        if isinstance(py, (bytes, unicode)):
            for i,C in enumerate(V['value.choices'] or self._choices):
                if py==C:
                    V['value.index'] = i
                    return

        # attempt to parse as integer
        V['value.index'] = py
