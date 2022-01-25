import sys
import unittest

if sys.version_info<(3,4) or sys.version_info>=(3,10):
    class TestDummy(unittest.TestCase):
        def test_asyncio(self):
            raise unittest.SkipTest("asyncio needs Py >=3.4.  >=3.10 not yet supported")

else:
    from .asynciotest import *
