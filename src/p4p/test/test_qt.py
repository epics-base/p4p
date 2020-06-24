import sys
import unittest

try:
    from qtpy.QtCore import QObject, QCoreApplication
except ImportError as e:
    class TestDummy(unittest.TestCase):
        def test_asyncio(self):
            raise unittest.SkipTest("Missing qtpy")

else:
    from .qttest import *
