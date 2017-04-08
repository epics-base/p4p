
from __future__ import print_function

import weakref, gc
import unittest

from .._p4p import Type as _Type

class TestRawType(unittest.TestCase):
    def testScalar(self):
        L = [
            ('a', 'i'),
            ('b', 'f'),
        ]
        T = _Type(spec=L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

    def testID(self):
        L = [
            ('a', 'i'),
            ('b', 'f'),
        ]
        T = _Type(spec=L, id='foo')

        self.assertEqual(T.aspy(),
                         ('s', 'foo', L))

    def testSubStruct(self):
        L = [
            ('a', 'i'),
            ('X', ('s', 'bar', [
                ('m', 's'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

    def testStructArray(self):
        L = [
            ('a', 'i'),
            ('X', ('as', 'bar', [
                ('m', 's'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

    def testUnion(self):
        L = [
            ('a', 'i'),
            ('X', ('u', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

    def testStructArray(self):
        L = [
            ('a', 'i'),
            ('X', ('au', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

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
            ('sub', ('s', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('asub', ('as', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('union', ('u', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('aunion', ('au', 'bar', [
                ('m', 's'),
                ('n', 'i'),
            ])),
            ('b', 'f'),
        ]
        T = _Type(L)

        self.assertEqual(T.aspy(),
                         ('s', 'structure', L))

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
