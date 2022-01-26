import sys
import unittest

if sys.version_info<(3,7):
    class TestDummy(unittest.TestCase):
        def test_asyncio(self):
            raise unittest.SkipTest("asyncio needs Py >=3.7")

else:
    from .asynciotest import *
