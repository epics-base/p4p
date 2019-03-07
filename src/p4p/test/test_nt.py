from __future__ import print_function

import logging
import sys
import unittest

from functools import partial
from collections import OrderedDict

from ..wrapper import Value, Type
from .. import nt
from .utils import RefTestCase

import numpy
from numpy.testing import assert_array_almost_equal as assert_aequal

_log = logging.getLogger(__name__)

class TestScalar(RefTestCase):

    def test_float_wrap(self, code='d', value=5.0):
        NT = nt.NTScalar(code)

        V = NT.wrap(value)
        self.assertEqual(V.value, value)
        self.assertEqual(V.alarm.severity, 0)
        self.assertIsNone(V.get('display'))

        NT = nt.NTScalar(code, display=True)
        V = NT.wrap({
            'value': value,
            'alarm': {
                'severity': 1,
            },
        })

        self.assertEqual(V.value, value)
        self.assertEqual(V.alarm.severity, 1)
        if code!='s':
            self.assertEqual(V.display.tolist(), [
                ('limitLow', 0.0),
                ('limitHigh', 0.0),
                ('description', u''),
                ('format', u''),
                ('units', u'')
            ])

    def test_int_wrap(self):
        self.test_float_wrap(code='i', value=42)
    def test_str_wrap(self):
        self.test_float_wrap(code='s', value='foo')

    def test_float_unwrap(self, code='d', value=5.0):
        NT = nt.NTScalar(code)
        V = NT.wrap({
            'value': value,
            'alarm': {
                'severity': 1,
            },
        })

        P = nt.NTScalar.unwrap(V)
        self.assertEqual(P, value)
        self.assertEqual(P.severity, 1)

        V2 = NT.wrap(P)
        self.assertIs(V, V2)

    def test_int_unwrap(self):
        self.test_float_unwrap(code='i', value=42)
    def test_str_unwrap(self):
        self.test_float_unwrap(code='s', value='foo')

    def test_array_wrap(self):
        NT = nt.NTScalar('ad')  # array of double

        A = numpy.asarray([1.0, 5.0])
        V = NT.wrap(A)
        assert_aequal(V.value, A)
        self.assertEqual(V.alarm.severity, 0)
        self.assertTrue(V.changed('value'))

    def test_array_unwrap(self):
        NT = nt.NTScalar('ad')  # array of double
        A = numpy.asarray(range(10))[2:5]
        V = NT.wrap(A)

        P = nt.NTScalar.unwrap(V)
        self.assertIsNot(P, A)
        assert_aequal(P, A)
        self.assertEqual(P.severity, 0)

    def test_string_array_wrap(self):
        NT = nt.NTScalar('as')  # array of string

        A = ["hello", "world"]
        V = NT.wrap(A)
        self.assertEqual(V.value, A)
        self.assertEqual(V.alarm.severity, 0)
        self.assertTrue(V.changed('value'))


class TestTable(RefTestCase):

    def test_wrap(self):
        NT = nt.NTTable(columns=[
            ('a', 'i'),
            ('b', 's'),
        ])
        V = NT.wrap([
            {'a': 5, 'b': 'one'},
            {'a': 6, 'b': 'two'},
        ])

        assert_aequal(V.value.a, [5, 6])
        self.assertEqual(V.value.b, ['one', 'two'])

    def test_unwrap(self):
        T = nt.NTTable.buildType(columns=[
            ('a', 'ai'),
            ('b', 'as'),
        ])
        V = Value(T, {
            'labels': ['a', 'b'],
            'value': {
                'a': [5, 6],
                'b': ['one', 'two'],
            },
        })

        P = list(nt.NTTable.unwrap(V))

        self.assertListEqual(P, [
            OrderedDict([('a', 5), ('b', u'one')]),
            OrderedDict([('a', 6), ('b', u'two')]),
        ])


class TestURI(RefTestCase):

    def test_build(self):
        NT = nt.NTURI([
            ('a', 'I'),
            ('b', 's'),
            ('c', ('S', None, [
                ('x', 'd'),
                ('y', 'd'),
            ])),
        ])

        V = NT.wrap('fn', (5,))
        self.assertEqual(V.query.a, 5)
        self.assertRaises(AttributeError, lambda: V.query.b)
        self.assertRaises(AttributeError, lambda: V.query.c)

        V = NT.wrap('fn', (6, 'foo'))
        self.assertEqual(V.query.a, 6)
        self.assertEqual(V.query.b, 'foo')
        self.assertRaises(AttributeError, lambda: V.query.c)

        V = NT.wrap('fn', (7,), {'b': 'bar'})
        self.assertEqual(V.query.a, 7)
        self.assertEqual(V.query.b, 'bar')
        self.assertRaises(AttributeError, lambda: V.query.c)

        V = NT.wrap('fn', (), {'a': 8, 'b': 'bar'})
        self.assertEqual(V.query.a, 8)
        self.assertEqual(V.query.b, 'bar')
        self.assertRaises(AttributeError, lambda: V.query.c)

        V = NT.wrap('fn', (), {'a': 8, 'b': 'bar', 'c': {'x': 1, 'y': 2}})
        self.assertEqual(V.query.a, 8)
        self.assertEqual(V.query.c.x, 1)
        self.assertEqual(V.query.c.y, 2)


class TestEnum(RefTestCase):

    def testStore(self):
        T = nt.NTEnum.buildType()

        V = Value(T, {
            'value.choices': ['zero', 'one', 'two'],
        })

        V.value = 'one'

        self.assertEqual(V.value.index, 1)

        V.value = '2'

        self.assertEqual(V.value.index, 2)

        V.value = 1

        self.assertEqual(V.value.index, 1)

    def testStoreBad(self):
        V = Value(nt.NTEnum.buildType(), {
            'value.choices': ['zero', 'one', 'two'],
        })

        V.value.index = 42

        def fn():
            V.value = self
        self.assertRaises(TypeError, fn)

        self.assertEqual(V.value.index, 42)

        def fn():
            V.value = 'other'
        self.assertRaises(ValueError, fn)

        self.assertEqual(V.value.index, 42)

        if sys.version_info >= (3, 0):
            V.value.choices = []

            def fn():
                V.value = '1'
            self.assertWarns(UserWarning, fn)  # warns of empty choices

            self.assertEqual(V.value.index, 1)

    def testSubStore(self):
        V = Value(Type([
            ('a', nt.NTEnum.buildType()),
            ('b', nt.NTEnum.buildType()),
        ]), {
            'a.value.choices': ['A', 'B'],
            'b.value.choices': ['X', 'Y'],
        })

        V.a = {'value': 'B'}

        self.assertEqual(V.a.value.index, 1)
        self.assertEqual(V.b.value.index, 0)

    def testWrap(self):
        W = nt.NTEnum()
        V = W.wrap({'index':1, 'choices':['X','Y']})

        self.assertEqual(V.value.index, 1)
        self.assertEqual(V.value.choices, ['X','Y'])

        W = nt.NTEnum()
        V = W.wrap(0)

        self.assertEqual(V.value.index, 0)
        self.assertEqual(V.value.choices, [])

    def testAssign(self):
        W = nt.NTEnum()
        V = nt.NTEnum.buildType()()
        W.assign(V, 1)

        self.assertEqual(V.value.index, 1)
        self.assertEqual(V.value.choices, [])

        V.value.choices = ['A', 'B']
        W.assign(V, 'A')

        self.assertEqual(V.value.index, 0)
        self.assertEqual(V.value.choices, ['A', 'B'])

    def testUnwrap(self):
        W = nt.NTEnum()
        V = nt.NTEnum.buildType()()

        U = W.unwrap(V)
        self.assertEqual(U, 0)
        self.assertIsNone(U.choice)

        V.value.index = 1
        V.value.choices = ['A', 'B']
        U = W.unwrap(V)
        self.assertEqual(U, 1)
        self.assertEqual(U.choice, 'B')
        self.assertEqual(W._choices, ['A', 'B'])

class TestArray(RefTestCase):

    def test_unwrap_None(self):
        V = Value(nt.NTNDArray.buildType(), {})

        img = nt.NTNDArray.unwrap(V)

        self.assertIsInstance(img, numpy.ndarray)
        assert_aequal(img.shape, (0,))

    def test_zero_length(self):
        V = Value(nt.NTNDArray.buildType(), {
            'value': numpy.arange(0),
            'dimension': [
                {'size': 3}, # X, columns
                {'size': 0}, # Y, rows
            ],
            'attribute': [
                {'name': 'ColorMode', 'value': 0},
            ],
        })

        img = nt.NTNDArray.unwrap(V)

        self.assertIsInstance(img, numpy.ndarray)
        assert_aequal(img.shape, (0,3))

    def test_unwrap_mono(self):
        pixels = numpy.asarray([  # 2x3
            [0, 1, 2],
            [3, 4, 5],
        ])
        # check my understanding of numpy
        self.assertTupleEqual(pixels.shape, (2, 3)) # inner-most right (in a pixel loop)
        assert_aequal(pixels.flatten(), [0, 1, 2, 3, 4, 5]) # row major

        V = Value(nt.NTNDArray.buildType(), {
            'value': numpy.arange(6),
            'dimension': [
                {'size': 3}, # X, columns
                {'size': 2}, # Y, rows
            ],
            'attribute': [
                {'name': 'ColorMode', 'value': 0},
            ],
        })

        img = nt.NTNDArray.unwrap(V)

        self.assertEqual(img.shape, (2, 3))
        assert_aequal(img, pixels)

        V2 = nt.NTNDArray().wrap(img)

        assert_aequal(V.value, V2.value)
        self.assertEqual(V.dimension[0].size, V2.dimension[0].size)
        self.assertEqual(V.dimension[1].size, V2.dimension[1].size)

    def test_unwrap_3d(self):
        # scipy.misc.face().shape==(768, 1024, 3)
        # in AD world this is [3, 1024, 768) w/ RGB1

        # we test with 4x2 RGB1
        pixels = numpy.array([
            [(1,0,0), (2,0,0), (3,0,0), (4,0,0)],
            [(5,0,0), (6,0,0), (7,0,0), (8,0,0)],
        ], dtype='u1')
        self.assertEqual(pixels.shape, (2, 4, 3))

        # manually construct matching NTNDArray
        V = Value(nt.NTNDArray.buildType(), {
            'value': ('ubyteValue', pixels.flatten()),
            'dimension': [
                {'size': 3}, # "color"
                {'size': 4}, # X, columns
                {'size': 2}, # Y, rows
            ],
            'attribute': [
                {'name': 'ColorMode', 'value': 2},
            ],
        })
        self.assertEqual(V.value.dtype, pixels.dtype)

        img = nt.NTNDArray.unwrap(V)

        self.assertEqual(img.shape, (2, 4, 3))
        assert_aequal(img, pixels)
        self.assertEqual(img.dtype, pixels.dtype)
        self.assertDictEqual(img.attrib, {u'ColorMode':2}) # RGB1

        # round trip
        V2 = nt.NTNDArray().wrap(img)

        assert_aequal(V.value, V2.value)
        self.assertEqual(V.dimension[0].size, V2.dimension[0].size)
        self.assertEqual(V.dimension[1].size, V2.dimension[1].size)
        self.assertEqual(V.dimension[2].size, V2.dimension[2].size)
        self.assertEqual(V2.value.dtype, pixels.dtype)
        self.assertEqual(V2.attribute[0].name, u'ColorMode')
        self.assertEqual(V2.attribute[0].value, 2)

        # wrap up raw array  (no pixels.attrib)
        V2 = nt.NTNDArray().wrap(pixels)

        assert_aequal(V.value, V2.value)
        self.assertEqual(V.dimension[0].size, V2.dimension[0].size)
        self.assertEqual(V.dimension[1].size, V2.dimension[1].size)
        self.assertEqual(V.dimension[2].size, V2.dimension[2].size)
        self.assertEqual(V2.value.dtype, pixels.dtype)
        self.assertEqual(V2.attribute[0].name, u'ColorMode')
        self.assertEqual(V2.attribute[0].value, 2)
