from __future__ import print_function

import unittest

from functools import partial
from collections import OrderedDict

from ..wrapper import Value, Type
from .. import nt

import numpy
from numpy.testing import assert_array_almost_equal as assert_aequal

class TestScalar(unittest.TestCase):

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

    test_int_unwrap = partial(test_float_unwrap, code='i', value=42)
    test_str_unwrap = partial(test_float_unwrap, code='s', value='foo')

    def test_array_wrap(self):
        NT = nt.NTScalar('ad') # array of double

        A = numpy.asarray([1.0, 5.0])
        V = NT.wrap(A)
        assert_aequal(V.value, A)
        self.assertEqual(V.alarm.severity, 0)


    def test_array_unwrap(self):
        NT = nt.NTScalar('ad') # array of double
        A = numpy.asarray(range(10))[2:5]
        V = NT.wrap(A)

        P = nt.NTScalar.unwrap(V)
        self.assertIsNot(P, A)
        assert_aequal(P, A)
        self.assertEqual(P.severity, 0)


class TestTable(unittest.TestCase):
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

class TestURI(unittest.TestCase):
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
