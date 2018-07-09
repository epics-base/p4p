from __future__ import print_function

import sys
import unittest

from functools import partial
from collections import OrderedDict

from ..wrapper import Value, Type
from .. import nt
from .utils import RefTestCase

import numpy
from numpy.testing import assert_array_almost_equal as assert_aequal

class TestScalar(RefTestCase):
    def test_float_wrap(self, code='d', value=5.0):
        NT = nt.NTScalar(code)

        V = NT.wrap(value)
        self.assertEqual(V.value, value)
        self.assertEqual(V.alarm.severity, 0)
        self.assertIsNone(V.get('display'))

        NT = nt.NTScalar(code, display=True)
        V = NT.wrap({
            'value':value,
            'alarm':{
                'severity':1,
            },
        })
        self.assertEqual(V.value, value)
        self.assertEqual(V.alarm.severity, 1)
        self.assertEqual(V.display.tolist(), [
            ('limitLow', 0.0),
            ('limitHigh', 0.0),
            ('description', u''),
            ('format', u''),
            ('units', u'')
        ])

    test_int_wrap = partial(test_float_wrap, code='i', value=42)
    test_str_wrap = partial(test_float_wrap, code='s', value='foo')

    def test_float_unwrap(self, code='d', value=5.0):
        NT = nt.NTScalar(code)
        V = NT.wrap({
            'value':value,
            'alarm':{
                'severity':1,
            },
        })

        P = nt.NTScalar.unwrap(V)
        self.assertEqual(P, value)
        self.assertEqual(P.severity, 1)

        V2 = NT.wrap(P)
        self.assertIs(V, V2)

    test_int_unwrap = partial(test_float_unwrap, code='i', value=42)
    test_str_unwrap = partial(test_float_unwrap, code='s', value='foo')

    def test_array_wrap(self):
        NT = nt.NTScalar('ad') # array of double

        A = numpy.asarray([1.0, 5.0])
        V = NT.wrap(A)
        assert_aequal(V.value, A)
        self.assertEqual(V.alarm.severity, 0)
        self.assertTrue(V.changed('value'))


    def test_array_unwrap(self):
        NT = nt.NTScalar('ad') # array of double
        A = numpy.asarray(range(10))[2:5]
        V = NT.wrap(A)

        P = nt.NTScalar.unwrap(V)
        self.assertIsNot(P, A)
        assert_aequal(P, A)
        self.assertEqual(P.severity, 0)

    def test_string_array_wrap(self):
        NT = nt.NTScalar('as') # array of string

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
            {'a': 5, 'b':'one'},
            {'a': 6, 'b':'two'},
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
            'value':{
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
        self.assertRaises(AttributeError, lambda:V.query.b)
        self.assertRaises(AttributeError, lambda:V.query.c)

        V = NT.wrap('fn', (6, 'foo'))
        self.assertEqual(V.query.a, 6)
        self.assertEqual(V.query.b, 'foo')
        self.assertRaises(AttributeError, lambda:V.query.c)

        V = NT.wrap('fn', (7,), {'b':'bar'})
        self.assertEqual(V.query.a, 7)
        self.assertEqual(V.query.b, 'bar')
        self.assertRaises(AttributeError, lambda:V.query.c)

        V = NT.wrap('fn', (), {'a':8, 'b':'bar'})
        self.assertEqual(V.query.a, 8)
        self.assertEqual(V.query.b, 'bar')
        self.assertRaises(AttributeError, lambda:V.query.c)

        V = NT.wrap('fn', (), {'a':8, 'b':'bar', 'c':{'x':1,'y':2}})
        self.assertEqual(V.query.a, 8)
        self.assertEqual(V.query.c.x, 1)
        self.assertEqual(V.query.c.y, 2)

class TestEnum(RefTestCase):
    def testStore(self):
        T = nt.NTEnum.buildType()

        V = Value(T, {
            'value.choices':['zero', 'one', 'two'],
        })

        V.value = 'one'

        self.assertEqual(V.value.index, 1)

        V.value = '2'

        self.assertEqual(V.value.index, 2)

        V.value = 1

        self.assertEqual(V.value.index, 1)

    def testStoreBad(self):
        V = Value(nt.NTEnum.buildType(), {
            'value.choices':['zero', 'one', 'two'],
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

        if sys.version_info>=(3,0):
            V.value.choices = []
            def fn():
                V.value = '1'
            self.assertWarns(UserWarning, fn) # warns of empty choices

            self.assertEqual(V.value.index, 1)

    def testSubStore(self):
        V = Value(Type([
            ('a', nt.NTEnum.buildType()),
            ('b', nt.NTEnum.buildType()),
        ]), {
            'a.value.choices':['A','B'],
            'b.value.choices':['X','Y'],
        })

        V.a = {'value':'B'}

        self.assertEqual(V.a.value.index, 1)
        self.assertEqual(V.b.value.index, 0)
