
import time
import numpy

from ..wrapper import Type, Value
from .common import alarm, timeStamp

from .scalar import ntwrappercommon

class NTNDArray(ntwrappercommon,numpy.ndarray):
    """
    Augmented numpy.ndarray with additional attributes

    * .attrib    - dictionary
    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """
    Value = Value

    attrib = None
    def _store(self, value):
        ntwrappercommon._store(self, value)
        self.attrib = {}
        for elem in value.get('attribute', []):
            self.attrib[elem.name] = elem.value

            if elem.name=='ColorMode' and elem.value!=0:
                raise ValueError("I only know about ColorMode gray scale, not mode=%d"%elem.value)

        shape = [D.size for D in value.dimension]
        shape.reverse()

        # in-place reshape!  Isn't numpy fun
        self.shape = shape

        return self

    @staticmethod
    def buildType(extra=[]):
        """Build type
        """
        return Type([
            ('value', 'v'),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
            ('dimension', ('S', None, [
                ('size', 'i'),
            ])),
            ('attribute', ('S', None, [
                ('name', 's'),
                ('value', 'v'),
            ])),
        ], id='epics:nt/NTNDArray:1.0')

    def __init__(self, **kws):
        self.type = self.buildType(**kws)

    #def wrap(self, value):
        #S, NS = divmod(time.time(), 1.0)
        #return Value(self.type, {
            #'value': A.ravel(),
            #'timeStamp': {
                #'secondsPastEpoch': S,
                #'nanoseconds': NS*1e9,
            #},
            #'attribute': [{'name':K, 'value':V} for K,V in value.attrib or {}],
            #'dimension': [{'size':N} for N in value.shape],
        #})

    @classmethod
    def unwrap(klass, value):
        """Unwrap Value as NTNDArray
        """
        return value.value.view(klass)._store(value)
