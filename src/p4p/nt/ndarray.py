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
from .common import alarm, timeStamp, NTBase

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

    Keys in the ``attrib`` dictionary may be any python which may be stored in a PVA field,
    including an arbitrary ``Value``.
    However, special handling is attempted if the provided ``Value`` appears to be an NTScalar
    or similar, in which case the .value, .alarm and .timeStamp are unpacked to the NTAttribute
    and other fields are discarded.
    """

    def __init__(self, *args, **kws):
        super(ntndarray, self).__init__(*args, **kws)
        self.attrib = {}

    def _store(self, value):
        ntwrappercommon._store(self, value)
        self.attrib = {}
        for elem in value.get('attribute', []):
            self.attrib[elem.name] = elem.value

        # we will fail if dimension[] contains None s, or if
        # the advertised shape isn't consistent with the pixel array length.
        shape = [D.size for D in value.dimension]
        shape.reverse()

        # in-place reshape!  Isn't numpy fun
        self.shape = shape or [0] # can't reshape if 0-d, so treat as 1-d if no dimensions provided

        return self


class NTNDArray(NTBase):
    """Representation of an N-dimensional array with meta-data

    Translates into `ntndarray`
    """
    Value = Value
    ntndarray = ntndarray

    # map numpy.dtype.char to .value union member name
    _code2u = {
        '?':'booleanValue',
        'b':'byteValue',
        'h':'shortValue',
        'i':'intValue',
        'l':'longValue',
        'B':'ubyteValue',
        'H':'ushortValue',
        'I':'uintValue',
        'L':'ulongValue',
        'f':'floatValue',
        'd':'doubleValue',
    }

    @classmethod
    def buildType(klass, extra=[]):
        """Build type
        """
        ret = klass._default_type
        if extra:
            L = ret.aspy()
            L.extend(extra)
            ret = Type(L, ret.getID())
        return ret

    _default_type = Type([
            ('value', ('U', None, [
                ('booleanValue', 'a?'),
                ('byteValue', 'ab'),
                ('shortValue', 'ah'),
                ('intValue', 'ai'),
                ('longValue', 'al'),
                ('ubyteValue', 'aB'),
                ('ushortValue', 'aH'),
                ('uintValue', 'aI'),
                ('ulongValue', 'aL'),
                ('floatValue', 'af'),
                ('doubleValue', 'ad'),
            ])),
            ('codec', ('S', 'codec_t', [
                ('name', 's'),
                ('parameters', 'v'),
            ])),
            ('compressedSize', 'l'),
            ('uncompressedSize', 'l'),
            ('uniqueId', 'i'),
            ('dataTimeStamp', timeStamp),
            ('alarm', alarm),
            ('timeStamp', timeStamp),
            ('dimension', ('aS', 'dimension_t', [
                ('size', 'i'),
                ('offset', 'i'),
                ('fullSize', 'i'),
                ('binning', 'i'),
                ('reverse', '?'),
            ])),
            ('attribute', ('aS', 'epics:nt/NTAttribute:1.0', [
                ('name', 's'),
                ('value', 'v'),
                ('tags', 'as'),
                ('descriptor', 's'),
                ('alarm', alarm),
                ('timeStamp', timeStamp),
                ('sourceType', 'i'),
                ('source', 's'),
            ])),
        ], id='epics:nt/NTNDArray:1.0')

    def __init__(self, **kws):
        self.type = self.buildType(**kws)

    def wrap(self, value, **kws):
        """Wrap numpy.ndarray as Value
        """
        attrib = getattr(value, 'attrib', None) or kws.pop('attrib', None) or {}

        value = numpy.asarray(value) # loses any special/augmented attributes
        dims = value.shape

        if 'ColorMode' not in attrib:
            # attempt to infer color mode from shape
            if value.ndim==2:
                attrib['ColorMode'] = 0 # gray

            elif value.ndim==3:
                for idx,dim in enumerate(reversed(dims)): # inner-most sent as left
                    if dim==3: # assume it's a color
                        attrib['ColorMode'] = 2 + idx  # 2 - RGB1, 3 - RGB2, 4 - RGB3
                        break # assume that the first is color, and any subsequent dim=3 is a thin ROI
                else:
                    raise ValueError("Unable to deduce color dimension from shape %r"%dims)

        dataSize = value.nbytes

        return self._annotate(Value(self.type, {
            'value': (self._code2u[value.dtype.char], value.flatten()),
            'compressedSize': dataSize,
            'uncompressedSize': dataSize,
            'uniqueId': 0,
            'attribute': [translateNDAttribute(K,V) for K, V in attrib.items()],
            'dimension': [{'size': N,
                           'offset': 0,
                           'fullSize': N,
                           'binning': 1,
                           'reverse': False} for N in reversed(dims)],
        }), **kws)

    @classmethod
    def unwrap(klass, value):
        """Unwrap Value as NTNDArray
        """
        V = value.value
        if V is None:
            # Union empty.  treat as zero-length char array
            V = numpy.zeros((0,), dtype=numpy.uint8)
        return V.view(klass.ntndarray)._store(value)

    def assign(self, V, py):
        """Store python value in Value
        """
        V[None] = self.wrap(py)

def translateNDAttribute(name, value):
    if isinstance(value, Value) and 'value' in value: # assume to be NT-like
        V = {
            'name': name,
            'value': value['value'],
        }
        if 'alarm' in value:
            V['alarm'] = value['alarm']
        if 'timeStamp' in value:
            V['timeStamp'] = value['timeStamp']
        return V

    return {'name': name, 'value': value}
