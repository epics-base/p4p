
from __future__ import print_function

import weakref, gc
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aequal

from .._p4p import (Type as _Type, Value as _Value)

class TestRawValue(unittest.TestCase):
    def testScalar(self):
        V = _Value(_Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival':42,
            'dval':4.2,
            'sval':'hello',
        })

        self.assertListEqual(V.tolist(), [
            ('ival', 42),
            ('dval', 4.2),
            ('sval', u'hello'),
        ])

        self.assertEqual(V.ival, 42)
        self.assertEqual(V.dval, 4.2)
        self.assertEqual(V.sval, u'hello')

        self.assertEqual(V['ival'], 42)
        self.assertEqual(V['dval'], 4.2)
        self.assertEqual(V['sval'], u'hello')

        V.ival = 43
        self.assertEqual(V.ival, 43)
        self.assertEqual(V['ival'], 43)

        V.dval = 4.3
        self.assertEqual(V.dval, 4.3)
        self.assertEqual(V['dval'], 4.3)

        V.sval = u'world'
        self.assertEqual(V.sval, u'world')
        self.assertEqual(V['sval'], u'world')

    def testArray(self):
        V = _Value(_Type([
            ('ival', 'ai'),
            ('dval', 'ad'),
            ('sval', 'as'),
        ]), {
            'ival': [1, 2, 3],
            'dval': np.asfarray([1.1, 2.2]),
            'sval': ['a', u'b'],
        })

        assert_aequal(V.ival, np.asarray([1, 2, 3]))
        assert_aequal(V.dval, np.asfarray([1.1, 2.2]))
        self.assertListEqual(V.sval, [u'a', u'b'])

    def testSubStruct(self):
        V = _Value(_Type([
            ('ival', 'i'),
            ('str', ('s', 'foo', [
                ('a', 'i'),
                ('b', 'i'),
            ])),
        ]), {
            'ival':42,
            'str':{
                'a': 1,
                'b': 2,
            },
        })

        self.assertListEqual(V.tolist(), [
            ('ival', 42),
            ('str', [
                ('a', 1),
                ('b', 2),
            ]),
        ])

        self.assertListEqual(V.tolist('str'), [
            ('a', 1),
            ('b', 2),
        ])

        self.assertEqual(V.str.a, 1)
        self.assertEqual(V.str['a'], 1)
        self.assertEqual(V['str'].a, 1)
        self.assertEqual(V['str']['a'], 1)
        self.assertEqual(V['str.a'], 1)

    def testVariantUnion(self):
        V = _Value(_Type([
            ('x', 'v'),
        ]))

        self.assertIsNone(V.x)

        V.x = 5
        self.assertEqual(V.x, 5)

        V.x = 1.1
        self.assertEqual(V.x, 1.1)

        V.x = 'test'
        self.assertEqual(V.x, u'test')

        V.x = u'test'
        self.assertEqual(V.x, u'test')

        V.x = np.asarray([1, 2])
        assert_aequal(V.x, np.asarray([1,2]))

        V.x = np.asfarray([1, 2])
        assert_aequal(V.x, np.asfarray([1,2]))

        #TODO: PVD bugs prevent this from working
        #V.x = None
        #self.assertIsNone(V.x)

    def testDisUnion(self):
        V = _Value(_Type([
            ('x', ('u', 'x', [
                ('a', 'i'),
                ('b', 's'),
            ])),
        ]))

        self.assertIsNone(V.x)

        V.x = ('a', 44)
        self.assertEqual(V.x, 44)

        V.x = ('b', 'test')
        self.assertEqual(V.x, u'test')

        # uses previously selected 'b'
        V.x = 'world'
        self.assertEqual(V.x, u'world')

        # uses previously selected 'b'
        V.x = 128;
        self.assertEqual(V.x, u'128')

        #TODO: PVD bugs prevent this from working
        #V.x = ('a', None) # another way to clear
        #self.assertIsNone(V.x)

    def testUnionArray(self):
        V = _Value(_Type([
            ('x', 'av'),
            ('y', ('au', 'foo', [
                ('a', 'i'),
                ('b', 's'),
            ])),
        ]))

        self.assertListEqual(V.x, [])
        self.assertListEqual(V.y, [])

        V.x = [1, 4.2, 'foo']
        V.y = [2, 5.2, 'bar']

        self.assertListEqual(V.x, [1, 4.2, u'foo'])
        # magic field selection strikes to convert float -> int
        self.assertListEqual(V.y, [2, 5, u'bar'])

        V.y = [('a', 3),
               ('b', 4.2),
               ('b', 'baz')]

        self.assertListEqual(V.y, [3, u'4.2', u'baz'])

        # union array assignment ignores previous selections
        V.y = [2, 5.2, 'bar']
        self.assertListEqual(V.y, [2, 5, u'bar'])
