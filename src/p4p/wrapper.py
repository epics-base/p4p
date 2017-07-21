
from ._p4p import Type, Value

def Struct(spec=None, id=None):
    return ('S', id, spec)

def Union(spec=None, id=None):
    return ('U', id, spec)

def StructArray(spec=None, id=None):
    return ('aS', id, spec)

def UnionArray(spec=None, id=None):
    return ('aU', id, spec)
