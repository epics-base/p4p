
import atexit

from ._p4p import Type, Value
from ._p4p import clearProviders

atexit.register(clearProviders)
