
from __future__ import print_function

import weakref
import gc
import unittest

from ..wrapper import Type as _Type
from .utils import RefTestCase


class TestRawType(RefTestCase):

    def testScalar(self):
        L = [
            ('a', 'i'),
            ('b', 'f'),
        ]
        T = _Type(spec=L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))
        self.assertEqual(T.aspy('a'), 'i')
        self.assertEqual(T['a'], 'i')
        self.assertEqual(len(T), 2)

    def testScalarTest(self):
        L = [
            ('a', 'i'),
            ('b', 'f'),
        ]
        T = _Type(spec=L)

        self.assertTrue(T.has('a'))
        self.assertTrue(T.has('b'))
        self.assertFalse(T.has('c'))
        self.assertListEqual(T.keys(), ['a', 'b'])

        self.assertTrue('a' in T)
        self.assertTrue('b' in T)
        self.assertFalse('c' in T)

    def testID(self):
        L = [
            ('a', 'i'),
            ('b', 'f'),
        ]
        T = _Type(spec=L, id='foo')

        self.assertEqual(T.aspy(),
                         ('S', 'foo', L))

    def testSubStruct(self):
        L = [
            ('a', 'i'),
            ('X', ('S', 'bar', [
                ('m', 's'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))
        self.assertEqual(T['X'].getID(), 'bar')
        self.assertEqual(T['X'].aspy(), T.aspy('X'))
        self.assertEqual(T['X'].aspy(), L[1][1])

    def testUnion(self):
        L = [
            ('a', 'i'),
            ('X', ('U', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))

    def testStructArray(self):
        L = [
            ('a', 'i'),
            ('X', ('aS', 'bar', [
                ('m', 's'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))

    def testStructArray2(self):
        L = [
            ('a', 'i'),
            ('X', ('aU', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))

    def testAll(self):
        L = [
            ('bool', '?'),
            ('str', 's'),
            ('i8', 'b'),
            ('u8', 'B'),
            ('i16', 'h'),
            ('u16', 'H'),
            ('i32', 'i'),
            ('u32', 'I'),
            ('i64', 'l'),
            ('u64', 'L'),
            ('f32', 'f'),
            ('f64', 'd'),
            ('any', 'v'),
            ('abool', 'a?'),
            ('astr', 'as'),
            ('ai8', 'ab'),
            ('au8', 'aB'),
            ('ai16', 'ah'),
            ('au16', 'aH'),
            ('ai32', 'ai'),
            ('au32', 'aI'),
            ('ai64', 'al'),
            ('au64', 'aL'),
            ('af32', 'af'),
            ('af64', 'ad'),
            ('aany', 'av'),
            ('sub', ('S', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('asub', ('aS', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('union', ('U', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('aunion', ('aU', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))

    def testReserved(self):
        L = [
            ("name", "as"),
            ("field", "aI"),
            ("type", "as"),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('S', 'structure', L))

    def testGC(self):
        T = _Type([('a', 'I')])
        self.assertTrue(gc.is_tracked(T))

        R = weakref.ref(T)
        del T
        gc.collect()
        T = R()
        if T is not None:
            print("Not Dead!", T, gc.get_referrers(T))
        self.assertIsNone(T)

    def testStructID(self):
        T = _Type([('a', 'I')])
        self.assertEqual(T.getID(), "structure")

        T = _Type([('a', 'I')], id="foo")
        self.assertEqual(T.getID(), "foo")

    def testExtend(self):
        B = _Type([('a', 'I')])
        S = _Type([('b', 'I')], base=B)

        self.assertTupleEqual(S.aspy(), ('S', 'structure', [
            ('a', 'I'),
            ('b', 'I'),
        ]))
