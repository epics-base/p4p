
import logging
import sys
_log = logging.getLogger(__name__)

try:
    from itertools import izip
except ImportError:
    izip = zip

import time
from collections import OrderedDict
from operator import itemgetter
from ..wrapper import Type, Value
from .common import timeStamp, alarm
from .scalar import NTScalar
from .ndarray import NTNDArray
from .enum import NTEnum

__all__ = [
    'NTScalar',
    'NTEnum',
    'NTMultiChannel',
    'NTTable',
    'NTNDArray',
]

_default_unwrap = {
    "epics:nt/NTScalar:1.0": NTScalar.unwrap,
    "epics:nt/NTScalarArray:1.0": NTScalar.unwrap,
    "epics:nt/NTNDArray:1.0": NTNDArray.unwrap,
}
_default_wrap = {
    "epics:nt/NTScalar:1.0": NTScalar.wrap,
    "epics:nt/NTScalarArray:1.0": NTScalar.wrap,
}


class NTMultiChannel(object):

    """Describes a structure holding the equivalent of a number of NTScalar
    """
    Value = Value

    @staticmethod
    def buildType(valtype, extra=[]):
        """Build a Type

        :param str valtype: A type code to be used with the 'value' field.  Must be an array
        :param list extra: A list of tuples describing additional non-standard fields
        :returns: A :py:class:`Type`
        """
        assert valtype[:1] == 'a', 'valtype must be an array'
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
                    ] + extra)


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

        :param list columns: List of column names and types. eg [('colA', 'd')]
        :param list extra: A list of tuples describing additional non-standard fields
        :returns: A :py:class:`Type`
        """
        return Type(id="epics:nt/NTTable:1.0",
                    spec=[
                    ('labels', 'as'),
                    ('value', ('S', None, columns)),
                    ('descriptor', 's'),
                    ('alarm', alarm),
                    ('timeStamp', timeStamp),
                    ] + extra)

    def __init__(self, columns=[], extra=[]):
        self.labels = []
        C = []
        for col, type in columns:
            if type[0] == 'a':
                raise ValueError("NTTable column types may not be array")
            C.append((col, 'a' + type))
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
        if isinstance(values, Value):
            return values
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
                if len(V) == 0:
                    del cols[L]

            try:
                return self.Value(self.type, {
                    'labels': self.labels,
                    'value': cols,
                })
            except:
                _log.error("Failed to encode '%s' with %s", cols, self.labels)
                raise
        except:
            _log.exception("Failed to wrap: %s", values)
            raise

    @staticmethod
    def unwrap(value):
        """Iterate an NTTable

        :returns: An iterator yielding an OrderedDict for each column
        """
        if len(value.labels) == 0:
            return

        # build lists of column names, and value
        lbl, cols = [], []
        for cname, cval in value.value.items():
            lbl.append(cname)
            cols.append(cval)

        # zip together column arrays to iterate over rows
        for rval in izip(*cols):
            # zip together column names and row values
            yield OrderedDict(zip(lbl, rval))


class NTURI(object):

    @staticmethod
    def buildType(args):
        """Build NTURI

        :param list args: A list of tuples of query argument name and PVD type code.

        >>> I = NTURI([
            ('arg_a', 'I'),
            ('arg_two', 's'),
        ])
        """
        try:
            return Type(id="epics:nt/NTURI:1.0", spec=[
                ('scheme', 's'),
                ('authority', 's'),
                ('path', 's'),
                ('query', ('S', None, args)),
            ])
        except Exception as e:
            raise ValueError('Unable to build NTURI compatible type from %s' % args)

    def __init__(self, args):
        self._args = list(args)
        self.type = self.buildType(args)

    def wrap(self, path, args=(), kws={}, scheme='', authority=''):
        """Wrap argument values (tuple/list with optional dict) into Value

        :param str path: The PV name to which this call is made
        :param tuple args: Ordered arguments
        :param dict kws: Keyword arguments
        :rtype: Value
        """
        # build dict of argument name+value
        AV = {}
        AV.update([A for A in kws.items() if A[1] is not None])
        AV.update([(N, V) for (N, _T), V in zip(self._args, args)])

        # list of argument name+type tuples for which a value was provided
        AT = [A for A in self._args if A[0] in AV]

        T = self.buildType(AT)
        try:
            return Value(T, {
                'scheme': scheme,
                'authority': authority,
                'path': path,
                'query': AV,
            })
        except Exception as e:
            raise ValueError('Unable to initialize NTURI %s from %s using %s' % (AT, AV, self._args))
