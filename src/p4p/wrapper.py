
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

    def changedSet(self, expand=False):
        """
        :param bool expand: Whether to include compress/shorthand fields when entire sub-structures are marked as changed.
                            If True, then compress bits are expanded.  If false, then only leaf fields will be included.
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
        """
        return _Value.changedSet(self, expand)

    # TODO: deprecate
    asSet = changedSet

    def clear(self):
        self.mark(None, False)

_Value._magic(Value)
