
from __future__ import print_function

import weakref
import gc
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aequal

from ..wrapper import Type, Value
from .. import pvdVersion
from .utils import RefTestCase


class TestRawValue(RefTestCase):

    def testToString(self):
        V = Value(Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival': 42,
            'dval': 4.2,
            'sval': 'hello',
        })

        # attempt to normalize to avoid comparing whitespace
        def proc(s):
            return list([l.split() for l in s.splitlines()])

        self.assertListEqual(proc(str(V)),
                         proc('''structure
    int ival 42
    double dval 4.2
    string sval hello
'''))

    def testScalar(self):
        V = Value(Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival': 42,
            'dval': 4.2,
            'sval': 'hello',
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

    def testIntegerRange(self):
        # test limits of intger ranges
        V = Value(Type([
            ('i32', 'i'),
            ('u32', 'I'),
            ('i64', 'l'),
            ('u64', 'L'),
        ]), {
            'i32': 0x7fffffff,
            'u32': 0xffffffff,
            'i64': 0x7fffffffffffffff,
            'u64': 0xffffffffffffffff,
        })

        self.assertEqual(V.i32, 0x7fffffff)
        self.assertEqual(V.u32, 0xffffffff)
        self.assertEqual(V.i64, 0x7fffffffffffffff)
        self.assertEqual(V.u64, 0xffffffffffffffff)

        V.i32 = -0x80000000
        V.i64 = -0x8000000000000000

        self.assertEqual(V.i32, -0x80000000)
        self.assertEqual(V.i64, -0x8000000000000000)

        # test setting out of range
        # TODO: not great...

        V.i64 = 0x8000000000000000
        self.assertEqual(V.i64, -0x8000000000000000)

        with self.assertRaises(OverflowError):
            V.i64 = 0x10000000000000000
        with self.assertRaises(OverflowError):
            V.i64 = -0x8000000000000001
        with self.assertRaises(OverflowError):
            V.u64 = 0x10000000000000000

        V.i32 = 0x80000000
        self.assertEqual(V.i32, -0x80000000)

        V.i32 = 0x100000000
        self.assertEqual(V.i32, 0)

    def testFieldAccess(self):
        V = Value(Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ]), {
            'ival': 42,
            'dval': 4.2,
            'sval': 'hello',
        })

        self.assertEqual(100, V.get('foo', 100))
        self.assertIsNone(V.get('foo'))

        self.assertRaises(KeyError, V.__getitem__, 'foo')
        self.assertRaises(AttributeError, getattr, V, 'foo')

        self.assertRaises(KeyError, V.__setitem__, 'foo', 5)
        self.assertRaises(AttributeError, setattr, V, 'foo', 5)

    def testReserved(self):
        L = [
            ("name", "s"),
            ("field", "I"),
            ("type", "s"),
        ]
        T = Type(L)
        V = Value(T)

        # item access works as always
        V['type'] = 'hello'
        self.assertEqual(V['type'], 'hello')

        # 'type' field hidden by method
        T2 = V.type() # ensure Value.type() isn't hidden

        # can't prevent overwriting the method...
        V.type = 4

    def testBadField(self):
        T = Type([
            ('ival', 'i'),
            ('dval', 'd'),
            ('sval', 's'),
        ])
        self.assertRaises(KeyError, Value, T, {'invalid': 42})

    def testArray(self):
        V = Value(Type([
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
        V = Value(Type([
            ('ival', 'i'),
            ('str', ('S', 'foo', [
                ('a', 'i'),
                ('b', 'i'),
            ])),
        ]), {
            'ival': 42,
            'str': {
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

        self.assertListEqual(V.keys(), ['ival', 'str'])
        self.assertTrue('ival' in V)
        self.assertTrue('str' in V)
        self.assertFalse('missing' in V)

    def testVariantUnion(self):
        V = Value(Type([
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
        assert_aequal(V.x, np.asarray([1, 2]))

        V.x = np.asfarray([1, 2])
        assert_aequal(V.x, np.asfarray([1, 2]))

        # clearing unions is broken prior to 7.0.0
        if pvdVersion() >= (7, 0, 0, 0):
            V.x = None
            self.assertIsNone(V.x)

    def testDisUnion(self):
        V = Value(Type([
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
        V.x = 128
        self.assertEqual(V.x, u'128')

        # clearing unions is broken prior to 7.0.0
        if pvdVersion() >= (7, 0, 0, 0):
            V.x = None  # another way to clear
            self.assertIsNone(V.x)

    def testUnionArray(self):
        V = Value(Type([
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
        S = Value(Type([
            ('x', 'i'),
        ]), {
            'x': 42,
        })

        V = Value(Type([
            ('y', 'av'),
        ]), {
            'y': [S],
        })

        # attribute/item access returns union array w/ struct as list of Value
        X = V.y

        self.assertIsInstance(X, list)
        self.assertIsInstance(X[0], Value)
        self.assertEqual(len(X), 1)

        # returns union array w/ struct as list of list of tuples
        Y = V.tolist()

        self.assertListEqual(Y, [
            ('y', [[
                ('x', 42),
            ]]),
        ])

    def testStructID(self):
        V = Value(Type([('a', 'I')]))
        self.assertEqual(V.getID(), "structure")

        V = Value(Type([('a', 'I')], id="foo"))
        self.assertEqual(V.getID(), "foo")

    def testStructArr(self):
        V = Value(Type([
            ('a', ('aS', None, [
                ('b', 'i'),
            ])),
        ]), {
            'a': [
                {'b': 4},
                [('b', 5)],
            ],
        })

        self.assertEqual(len(V.a), 2)
        self.assertEqual(V.a[0].b, 4)
        self.assertEqual(V.a[1].b, 5)

        V.a = [{'b': 1}]

        self.assertEqual(len(V.a), 1)
        self.assertEqual(V.a[0].b, 1)

    def testBitSet(self):
        A = Value(Type([
            ('x', 'i'),
            ('y', 'i'),
        ]), {
            'y': 42,
        })
        # initially all un-"changed" (at default)

        self.assertSetEqual(A.changedSet(), {'y'})
        self.assertTrue(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertRaises(KeyError, A.changed, 'invalid')

        A.mark('x')

        self.assertSetEqual(A.changedSet(), {'x', 'y'})
        self.assertTrue(A.changed())
        self.assertTrue(A.changed('x'))
        self.assertTrue(A.changed('y'))

        A.mark('x', False)

        self.assertSetEqual(A.changedSet(), {'y'})
        self.assertTrue(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))

        A.unmark()

        self.assertSetEqual(A.changedSet(), set())
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertFalse(A.changed('y'))

    def testBitSetRecurse(self):
        A = Value(Type([
            ('x', 'i'),
            ('y', 'i'),
            ('z', ('S', None, [
                ('a', 'i'),
                ('b', 'i'),
            ])),
        ]), {
        })

        self.assertSetEqual(A.changedSet(), set())
        self.assertFalse(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertFalse(A.changed('y'))
        self.assertFalse(A.changed('z.a'))
        self.assertFalse(A.changed('z.b'))

        A.mark('y')
        A.mark('z.a')
        self.assertSetEqual(A.changedSet(), {'y', 'z.a'})
        self.assertTrue(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertTrue(A.changed('z.a'))
        self.assertFalse(A.changed('z.b'))

        Z = A.z  # A and Z share fields and bitset
        self.assertTrue(Z.changed('a'))
        self.assertFalse(Z.changed('b'))

        Z.mark('a', False)
        Z.mark('b', True)

        self.assertTrue(A.changed())
        self.assertFalse(A.changed('x'))
        self.assertTrue(A.changed('y'))
        self.assertFalse(A.changed('z.a'))
        self.assertTrue(A.changed('z.b'))
        self.assertFalse(Z.changed('a'))
        self.assertTrue(Z.changed('b'))

    def testBitSetSubStruct(self):
        A = Value(Type([
            ('x', 'i'),
            ('y', 'i'),
            ('z', ('S', None, [
                ('a', 'i'),
                ('b', 'i'),
                ('q', ('S', None, [
                    ('m', 'i'),
                ])),
            ])),
        ]), {
        })

        self.assertSetEqual(A.changedSet(), set())
        A.mark('z')
        self.assertSetEqual(A.changedSet(expand=False, parents=False), {'z'})
        self.assertSetEqual(A.changedSet(expand=True, parents=False),  {     'z.a', 'z.b',        'z.q.m'})
        self.assertSetEqual(A.changedSet(expand=False, parents=True),  {'z'})
        self.assertSetEqual(A.changedSet(expand=True, parents=True),   {'z', 'z.a', 'z.b', 'z.q', 'z.q.m'})

        A.unmark()
        A.mark('z.a')
        self.assertSetEqual(A.changedSet(expand=False, parents=False), {     'z.a'})
        self.assertSetEqual(A.changedSet(expand=True, parents=False),  {     'z.a'})
        self.assertSetEqual(A.changedSet(expand=False, parents=True),  {'z', 'z.a'})
        self.assertSetEqual(A.changedSet(expand=True, parents=True),   {'z', 'z.a'})

        A.unmark()
        A.mark('z.q')
        self.assertSetEqual(A.changedSet(expand=False, parents=False), {     'z.q'})
        self.assertSetEqual(A.changedSet(expand=True, parents=False),  {            'z.q.m'})
        self.assertSetEqual(A.changedSet(expand=False, parents=True),  {'z', 'z.q'})
        self.assertSetEqual(A.changedSet(expand=True, parents=True),   {'z', 'z.q', 'z.q.m'})

    def testSubStructAssignment(self):
        A = Value(Type([
            ('x', 'i'),
            ('y', 'i'),
            ('z', ('S', None, [
                ('a', 'i'),
                ('b', 'i'),
                ('c', 'i'),
            ])),
            ('q', 'v'),
        ]), {
        })

        B = Value(Type([
            ('a', 'i'),
            ('b', 'i'),
        ]), {
            'a':42,
            'b':43,
        })

        A.z = B # copy alike structure (absense of 'c' is ok)
        A.q = B # variant union assignment (absense of 'c' also ok)

        self.assertEqual(A.z.a, 42)
        self.assertEqual(A.z.b, 43)
        self.assertEqual(A.q.a, 42)
        self.assertEqual(A.q.b, 43)
        self.assertSetEqual(A.changedSet(), {'z.a', 'z.b', 'q'})

class TestReInit(RefTestCase):

    def testCopySubStruct(self):
        A = Value(Type([
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
            'x.y': 42,
            'x.z': range(3),
            'x.q': 'hello',
            'x.m': ['hello', 52],
            'x.a.A': 100,
        })

        B = Value(A.type(), {
            'x.y': 43,
            'x.z': range(4),
            'x.q': 15,
            'x.m': ['world', 62],
            'x.a.A': 101,
        })

        print(A.x, B.x)
        B.x = A.x

        self.assertEqual(B.x.y, 42)
        assert_aequal(B.x.z, [0, 1, 2])
        self.assertEqual(B.x.q, 'hello')
        self.assertEqual(B.x.m, ['hello', 52])
        self.assertEqual(B.x.a.A, 100)
