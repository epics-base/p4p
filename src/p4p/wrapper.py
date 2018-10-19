# This module defines sub-classes of C extension classes
# which add functionality that is better expressed in python.
# These types are then pushed (by _magic) down into extension
# code where they will be used as the types passed to callbacks.
from ._p4p import (Type as _Type, Value as _Value)

__all__ = (
    'Type',
    'Value',
    'Struct',
    'StructArray',
    'Union',
    'UnionArray',
)


def Struct(spec=None, id=None):
    return ('S', id, spec)


def Union(spec=None, id=None):
    return ('U', id, spec)


def StructArray(spec=None, id=None):
    return ('aS', id, spec)


def UnionArray(spec=None, id=None):
    return ('aU', id, spec)


class Type(_Type):
    """Type([fields..., id=None])

    A definition of a data structure consisting of a list of field names and types,
    as well as an optional type name string (id="").
    Field type specifications are either a string eg. "d" (double precision float),
    or a tuple ("S", None, [fields...) defining a sub-structure. ::

        T = Type([
            ('value', 'I'),
        ])

    Defines a structure with a single field named 'value' with type u32 (unsigned integer width 32-bit).

    An example of defining a sub-structure. ::

        T = Type([
            ('value', ('S', None, [
                ('index', 'i'),
            ])),
        ])

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
    """
    __slots__ = []  # we don't allow custom attributes for now
    __contains__ = _Type.has

    def __iter__(self):
        for k in self.keys():
            yield k

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

    def __repr__(self):
        S, id, fields = self.aspy()
        return 'Type(%s, id="%s")' % (fields, id)
    __str__ = __repr__

_Type._magic(Type)


class Value(_Value):
    """Value(type[, initial])

    Representation of a data structure of a particular :py:class:`Type`.

    Can be created using a :py:class:`Type`, with an optional dict containing initial values. ::

        A = Value(Type([
            ('value', 'I'),
        ]), {
            'value': 42,
        })

    Defines a structure with a single field named 'value' with type u32 (unsigned integer width 32-bit).

    An example of defining a sub-structure. ::

        A = Value(Type([
            ('value', ('S', None, [
                ('index', 'i'),
            ])),
        ]), {
            'value': {'index', 5},
            # 'value.index': 5,  # equivalent
        })

    Defines a structure containing a sub-structure 'value' which has a single field 'index' which is
    a signed 32-bit integer.
    """
    __slots__ = []  # prevent attribute access to invalid fields

    __contains__ = _Value.has

    def keys(self):
        return self.type().keys()

    def __iter__(self):
        return iter(self.type())

    def changed(self, *fields):
        """Test if one or more fields have changed.

        A field is considered to have changed if it has been marked as changed,
        or if any of its parent, or child, fields have been marked as changed.
        """
        S = super(Value, self).changed
        for fld in fields or (None,): # no args tests for any change
            if S(fld):
                return True
        return False

    def changedSet(self, expand=False, parents=False):
        """
        :param bool expand: Whether to expand when entire sub-structures are marked as changed.
                            If True, then sub-structures are expanded and only leaf fields will be included.
                            If False, then a direct translation is made, which may include both leaf and sub-structure fields.
        :param bool parents: If True, include fake entries for parent sub-structures with leaf fields marked as changed.
        :returns: A :py:class:`set` of names of those fields marked as changed.

        Return a :py:class:`set` containing the names of all changed fields. ::

            A = Value(Type([
                ('x', 'i'),
                ('z', ('S', None, [
                    ('a', 'i'),
                    ('b', 'i'),
                ])),
            ]), {
            })

            A.mark('z')
            assert A.changedSet(expand=False) == {'z'}         # only shows fields explicitly marked
            assert A.changedSet(expand=True) == {'z.a', 'z.b'} # actually used during network transmission
            A.mark('z.a') # redundant
            assert A.changedSet(expand=False) == {'z', 'z.a'}
            assert A.changedSet(expand=True) == {'z.a', 'z.b'}
            A.unmark('z')
            assert A.changedSet(expand=False) == {'z.a'}
            assert A.changedSet(expand=True) == {'z.a'}
            assert A.changedSet(expand=False, parents=True) == {'z', 'z.a'}
            assert A.changedSet(expand=True, parents=True) == {'z', 'z.a'}


        * expand=False, parents=False gives a direct mapping of the underlying BitSet as it would (get/monitor),
          or have been (put/rpc), moved over the network.
        * expand=True, parents=False gives the effective set of leaf fields which will be moved over the network.
          taking into account the use of whole sub-structure compress/shorthand bits.
        * expand=False, parents=True gives a way of testing if anything changed within a set of interesting fields
          (cf. set.intersect).
        """
        return _Value.changedSet(self, expand, parents)

    # TODO: deprecate
    asSet = changedSet

    def clear(self):
        self.mark(None, False)

    def __str__(self):
        return self.tostr(limit=100)
    __repr__ = __str__

_Value._magic(Value)
