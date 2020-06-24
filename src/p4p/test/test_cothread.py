
import sys
import unittest

try:
    import cothread
except ImportError:
    class TestDummy(unittest.TestCase):
        def test_asyncio(self):
            raise unittest.SkipTest("Missing cothread")

else:
    from .cothreadtest import *
