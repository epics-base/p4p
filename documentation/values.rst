Working with Value and Type
===========================

.. currentmodule:: p4p

Working with Type and Value
---------------------------

:py:class:`Value` is initialized in two steps.
First a :py:class:`Type` describing the data structure is created,
then the Value container is build, and optionally initialized.

   >>> from p4p import Type, Value
   >>> T = Type([('value', 'i')])
   >>> V = Value(T, {'value':42})

Here a simple structure is defined with a single field 'value' which is a signed 32-bit integer.
The created value initializes 'value' to 42.
This Value can then be accessed with:

   >>> V.value
   42
   >>> V['value']
   42
   >>> V.get('value', 111)
   42
   >>> V.get('invalid', 111) # uses default
   111

Field values can also be changed

   >>> V.value = 43
   >>> V['value'] = 43

Type definitions
----------------

The :py:mod:`p4p.nt` module can be used to build common :py:class:`Type` definitions.

Structures are strongly typed.
The type is specified with a code.
Supported codes are given in the table below,
and may be prefixed with 'a' to make an array.
For example, 'ai' is an array of signed 32-bit integers.

Type specifier codes:

==== =======
Code Type
==== =======
?    bool
s    unicode
b    s8
B    u8
h    s16
H    u16
i    i32
I    u32
l    i64
L    u64
f    f32
d    f64
v    variant
U    union
S    struct
==== =======

A :py:class:`Type` is build with a list of tuples,
where each tuple defines a field.

For all type codes except struct 's' and discriminating union 'u' only the type code is needed.

   >>> T = Type([
      ('value', 's'), # string
      ('other', 'ad'), # array of double floating
   ])

sub-structures and discriminating union have a nested tuple to fully define the field type.

   >>> T = Type([
      ('value', 's'), # string
      ('alarm', ('S', None, [
          ('severity', 'i'),
          ('status', 'i'),
          ('message', 's'),
      ])),
   ])
   >>> V = Value(T, {'alarm':{'severity':0}})
   >>> V.alarm.severity
   0
   >>> V.alarm['severity']
   0
   >>> V['alarm.severity']
   0

Here a sub-structure 'alarm' is defined with three fields.

A discriminating union is defined in the same manner.

   >>> T = Type([
      ('value', ('u', None, [
          ('ival', 'i'),
          ('sval', 's'),
      ])),
   ])
   >>> V.value
   None
   >>> V.value = ('ival', 42) # explicitly select union field name
   >>> V.value
   42
   >>> V.value = ('sval', 'hello')
   >>> V.value
   u'hello'
   >>> V.value = 43   # beware still using 'sval' !!
   >>> V.value
   u'43'

Assigning variant and union
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the preceding example suggests, the rules for assigning values
to variant and union fields can be surprising.

The rules for assigning a variant are as follows:

* None - clears current value
* Value - Stores a structure
* int - signed 32-bit
* long - signed 64-bit
* float - 64-bit floating
* bytes|unicode - string
* ndarray of integer or floating - array of integer or floating

For a variant, other values throw an Exception.

The rules for assigning a discriminating union are as follows:

* None - clears current value
* ('field', val) - explicitly specify the union field name
* val - If a union field has previously been selected, coerce assigned value
* val - If no union field previously select, attempt magic selection and coerce.

An Exception is thrown otherwise.

API Reference
-------------

.. module:: p4p

.. autoclass:: Value

    .. automethod:: tolist

    .. automethod:: items

    .. automethod:: getID

    .. automethod:: type

    .. automethod:: get

    .. automethod:: select

    .. method:: __getattr__(field)

        Access a sub-field.  If the sub-field value.

    .. method:: __setattr__(field, value)

        Assign sub-field.

    .. method:: __getitem__(field)

        Same as __getattr__

    .. method:: __setitem__(field, value)

        Same as __setattr__

    .. automethod:: changed

    .. automethod:: mark

    .. automethod:: asSet

.. autoclass:: Type

    .. automethod:: getID

    .. automethod:: aspy

    .. automethod:: has

Relation to C++ API
-------------------

For those familiar with the pvDataCPP API.
A :py:class:`Type` wraps a Structure.
:py:class:`Value` wraps a
PVStructure and an associated BitSet describing which fields have been
initialized.
