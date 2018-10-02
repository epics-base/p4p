"""Helper for handling NTNDArray a la. areaDetector.

Known attributes

"ColorMode"   (inner-most left, as given in NDArray.cpp, numpy.ndarray.shape is reversed from this)
 0 - Mono   [Nx, Ny]
 1 - Bayer  [Nx, Ny]
 2 - RGB1   [3, Nx, Ny]
 3 - RGB2   [Nx, 3, Ny]
 4 - RGB3   [Nx, Ny, 3]
 5 - YUV444 ?
 6 - YUV422 ??
 7 - YUV411 ???
"""

import logging
_log = logging.getLogger(__name__)

import time
import numpy

from ..wrapper import Type, Value
from .common import alarm, timeStamp

from .scalar import ntwrappercommon


class ntndarray(ntwrappercommon, numpy.ndarray):

    """
    Augmented numpy.ndarray with additional attributes

    * .attrib    - dictionary
    * .severity
    * .status
    * .timestamp - Seconds since 1 Jan 1970 UTC as a float
    * .raw_stamp - A tuple (seconds, nanoseconds)
    * .raw - The underlying :py:class:`p4p.Value`.
    """

    attrib = None

    def _store(self, value):
        ntwrappercommon._store(self, value)
        self.attrib = {}
        for elem in value.get('attribute', []):
            self.attrib[elem.name] = elem.value

        shape = [D.size for D in value.dimension]
        shape.reverse()

        # in-place reshape!  Isn't numpy fun
        self.shape = shape

        return self


class NTNDArray(object):
    """Representation of an N-dimensional array with meta-data
    """
    Value = Value
    ntndarray = ntndarray

    @staticmethod
    def buildType(extra=[]):
        """Build type
        """
        return Type([
            ('value', 'v'),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
            ('dimension', ('aS', None, [
                ('size', 'i'),
            ])),
            ('attribute', ('aS', None, [
                ('name', 's'),
                ('value', 'v'),
            ])),
        ], id='epics:nt/NTNDArray:1.0')

    def __init__(self, **kws):
        self.type = self.buildType(**kws)

    def wrap(self, value):
        """Wrap numpy.ndarray as Value
        """
        S, NS = divmod(time.time(), 1.0)
        value = numpy.asarray(value)
        dims = list(value.shape)
        dims.reverse() # inner-most sent as left

        attrib = getattr(value, 'attrib', {})
        if 'ColorMode' not in attrib:
            attrib['ColorMode'] = 0 if value.ndim==2 else 4 # NDArray::getInfo() treats unknown as RGB3
        # else: assume caller knows what ColorMode means

        return Value(self.type, {
            'value': value.flatten(),
            'timeStamp': {
                'secondsPastEpoch': S,
                'nanoseconds': NS * 1e9,
            },
            'attribute': [{'name': K, 'value': V} for K, V in attrib.items()],
            'dimension': [{'size': N} for N in dims],
        })

    @classmethod
    def unwrap(klass, value):
        """Unwrap Value as NTNDArray
        """
        return value.value.view(klass.ntndarray)._store(value)
