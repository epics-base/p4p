
from __future__ import print_function

import weakref, gc
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aequal

from .._p4p import (Type as _Type, Value as _Value)
from ..wrapper import Value
from .. import pvdVersion

class TestRawValue(unittest.TestCase):
    def testToString(self):
        V = _Value(_Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival':42,
            'dval':4.2,
            'sval':'hello',
        })

        self.assertEqual(str(V),
                         '''structure 
    int ival 42
    double dval 4.2
    string sval hello
''')
        
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

        self.assertListEqual(V.items(), [
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

    def testFieldAccess(self):
        V = _Value(_Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival':42,
            'dval':4.2,
            'sval':'hello',
        })

        self.assertEqual(100, V.get('foo', 100))
        self.assertIsNone(V.get('foo'))

        self.assertRaises(KeyError, V.__getitem__, 'foo')
        self.assertRaises(AttributeError, getattr, V, 'foo')

        self.assertRaises(KeyError, V.__setitem__, 'foo', 5)
        self.assertRaises(AttributeError, setattr, V, 'foo', 5)

    def testBadField(self):
        T = _Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ])
        self.assertRaises(KeyError, _Value, T, {'invalid':42})

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
            ('str', ('S', 'foo', [
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

        self.assertDictEqual(V.todict(), {
            'ival': 42,
            'str': {
                'a': 1,
                'b': 2,
            },
        })

        self.assertListEqual(V.tolist('str'), [
            ('a', 1),
            ('b', 2),
        ])

        self.assertEqual(V.str.a, 1)
        self.assertEqual(V.str['a'], 1)
        self.assertEqual(V['str'].a, 1)
        self.assertEqual(V['str']['a'], 1)
        self.assertEqual(V['str.a'], 1)

        self.assertEqual(V.type().aspy(),
            ('S', 'structure', [
                ('ival', 'i'),
                ('str', ('S', 'foo', [
                    ('a', 'i'),
                    ('b', 'i'),
                ])),
            ]),
        )

        self.assertEqual(V.type('str').aspy(),
            ('S', 'foo', [
                ('a', 'i'),
                ('b', 'i'),
            ]),
        )

        self.assertRaises(KeyError, V.type, 'invalid')

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

        # clearing unions is broken prior to 7.0.0
        if pvdVersion()>=(7,0,0,0):
            V.x = None
            self.assertIsNone(V.x)

    def testDisUnion(self):
        V = _Value(_Type([
            ('x', ('U', 'x', [
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

        # clearing unions is broken prior to 7.0.0
        if pvdVersion()>=(7,0,0,0):
            V.x = None # another way to clear
            self.assertIsNone(V.x)

    def testUnionArray(self):
        V = _Value(_Type([
            ('x', 'av'),
            ('y', ('aU', 'foo', [
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

    def testUnionArrayStruct(self):
        S= _Value(_Type([
            ('x', 'i'),
        ]), {
            'x': 42,
        })

        V = _Value(_Type([
            ('y', 'av'),
        ]), {
            'y': [S],
        })

        # attribute/item access returns union array w/ struct as list of Value
        X = V.y

        self.assertIsInstance(X, list)
        self.assertIsInstance(X[0], _Value)
        self.assertEqual(len(X), 1)

        # returns union array w/ struct as list of list of tuples
        Y = V.tolist()

        self.assertListEqual(Y, [
            ('y', [[
                ('x', 42),
            ]]),
        ])

    def testStructID(self):
        V = Value(_Type([('a', 'I')]))
        self.assertEqual(V.getID(), "structure")

        V = Value(_Type([('a', 'I')], id="foo"))
        self.assertEqual(V.getID(), "foo")

    def testRepr(self):
        V = Value(_Type([('a', 'I')]))
        self.assertEqual(repr(V), 'Value(id:structure, a:0)')
        
        V = Value(_Type([('a', 'I'), ('value', 'd')]))
        self.assertEqual(repr(V), 'Value(id:structure, value:0.0)')

        V = Value(_Type([('a', 'I')], id='foo'))
        self.assertEqual(repr(V), 'Value(id:foo, a:0)')
        
        V = Value(_Type([('a', 'I'), ('value', 'd')], id='foo'))
        self.assertEqual(repr(V), 'Value(id:foo, value:0.0)')

    def testBitSet(self):
        A= _Value(_Type([
            ('x', 'i'),
            ('y', 'i'),
        ]), {
            'y': 42,
        })
        # initially all un-"changed" (at default)

        self.assertSetEqual(A.asSet(), {'y'})
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertRaises(KeyError, A.changed, 'invalid')

        A.mark('x')

        self.assertSetEqual(A.asSet(), {'x', 'y'})
        self.assertFalse(A.changed())
        self.assertTrue(A.changed('x'))
        self.assertTrue(A.changed('y'))

        A.mark('x', False)

        self.assertSetEqual(A.asSet(), {'y'})
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))

    def testBitSetRecurse(self):
        A= _Value(_Type([
            ('x', 'i'),
            ('y', 'i'),
            ('z', ('S', None, [
                ('a', 'i'),
                ('b', 'i'),
            ])),
        ]), {
        })

        self.assertSetEqual(A.asSet(), set())
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertFalse(A.changed('y'))
        self.assertFalse(A.changed('z.a'))
        self.assertFalse(A.changed('z.b'))

        A.mark('y')
        A.mark('z.a')
        self.assertSetEqual(A.asSet(), {'y', 'z.a'})
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertTrue(A.changed('z.a'))
        self.assertFalse(A.changed('z.b'))

        Z = A.z # A and Z share fields and bitset
        self.assertTrue(Z.changed('a'))
        self.assertFalse(Z.changed('b'))

        Z.mark('a', False)
        Z.mark('b', True)

        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertFalse(A.changed('z.a'))
        self.assertTrue(A.changed('z.b'))
        self.assertFalse(Z.changed('a'))
        self.assertTrue(Z.changed('b'))

class TestReInit(unittest.TestCase):
    def testCopySubStruct(self):
        A = _Value(_Type([
            ('x', ('S', None, [
                ('y', 'i'),
                ('z', 'ai'),
                ('q', 'v'),
                ('m', 'av'),
                ('a', ('S', None, [
                    ('A', 'i'),
                ])),
            ])),
        ]), {
            'x.y':42,
            'x.z':range(3),
            'x.q':'hello',
            'x.m':['hello', 52],
            'x.a.A':100,
        })

        B = _Value(A.type(), {
            'x.y':43,
            'x.z':range(4),
            'x.q':15,
            'x.m':['world', 62],
            'x.a.A':101,
        })

        print(A.x, B.x)
        B.x = A.x

        self.assertEqual(B.x.y, 42)
        assert_aequal(B.x.z, [0,1,2])
        self.assertEqual(B.x.q, 'hello')
        self.assertEqual(B.x.m, ['hello', 52])
        self.assertEqual(B.x.a.A, 100)
