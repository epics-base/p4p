
from __future__ import print_function

import logging
import sys
import gc
import inspect
import unittest
import time
import os
import tempfile
import fnmatch
import weakref

from functools import wraps

from .. import listRefs
from .._p4p import _forceLazy

_log = logging.getLogger(__name__)

_forceLazy()

if not hasattr(unittest.TestCase, 'assertRegex'):
    unittest.TestCase.assertRegex = unittest.TestCase.assertRegexpMatches

if not hasattr(unittest.TestCase, 'assertRaisesRegex'):
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

class RefTestMixin(object):
    __showLeftovers = True

    """Ensure that each test does not result in a net change in extension object counts
    """
    # set to list of names to compare.  Set to None to disable
    ref_check = ('*',)

    def __refs(self, refs=None):
        refs = refs or listRefs()
        _log.debug("REFS %s", refs)
        names = set()
        for pat in self.ref_check:
            names |= set(fnmatch.filter(refs, pat))
        return dict([(K, V) for K, V in refs.items() if K in names and V>0])

    def setUp(self):
        self.__traceme = set()
        if self.ref_check is not None:
            self.__before = self.__refs()

            for mustzero in ('ClientContextImpl',):
                if self.__before.get(mustzero, 0)!=0 and self.__showLeftovers:
                    self.__showLeftovers = False # only show failure once
                    self.fail('Leftovers from previous test: %s = %d'%(mustzero, self.__before[mustzero]))

        super(RefTestMixin, self).setUp()

    def traceme(self, obj):
        self.__traceme.add(weakref.ref(obj))

    def _sleep(self, delay):
        time.sleep(delay)

    def tearDown(self):
        super(RefTestMixin, self).tearDown()
        if self.ref_check is not None:
            traceme = list(self.__traceme)
            del self.__traceme
            gc.collect()
            after = self.__refs()

            test = self.__before == after

            for mustzero in ('ClientContextImpl',):
                test &= after.get(mustzero, 0)==0

            frame = inspect.currentframe()
            for T in traceme:
                O = T()
                if O is None:
                    continue
                nrefs = sys.getrefcount(O)
                refs = gc.get_referrers(O)
                nrefs -= len(refs) # exclude tracked refs
                refs = filter(lambda o:o not in (frame, traceme), refs)
                _log.debug("ALIVE %s -> %s + %d ext refs", O, refs, nrefs)

            self.assertDictEqual(self.__before, after)
            # check for any obviously corrupt counters, even those not being compared
            # self.assertFalse(any([V>1000000 for V in refs.values()]), "before %s after %s"%(self.__raw_before, refs))

            if not test:
                for mustzero in ('ClientContextImpl', 'ServerPvt'):
                    self.assertEqual(0, after.get(mustzero, 0), mustzero)
                self.assertDictEqual(self.__before, after)

class RefTestCase(RefTestMixin, unittest.TestCase):
    def __init__(self, methodName='runTest'):
        # skip reference check for tests which have already failed.
        meth = getattr(self, methodName)
        @wraps(meth)
        def wrapper(*args, **kws):
            try:
                return meth(*args, **kws)
            except:
                self.ref_check = None
                raise
        setattr(self, methodName, wrapper)
        super(RefTestCase, self).__init__(methodName=methodName)

    def setUp(self):
        super(RefTestCase, self).setUp()

    def tearDown(self):
        super(RefTestCase, self).tearDown()

    if not hasattr(unittest.TestCase, 'assertRegex'):
        def assertRegex(self, text, regex):
            import re
            self.assertTrue(re.search(regex, text),
                            """Regex didn't match: %r not found in %r"""%(regex, text))


def gctrace(obj, maxdepth=8):
    # depth first traversal
    pop = object()
    top = inspect.currentframe()
    next = top.f_back
    stack, todo = [], [obj]
    visited = set()

    while len(todo):
        obj = todo.pop(0)
        I = id(obj)
        if inspect.isframe(obj):
            S = 'Frame %s:%d' % (obj.f_code.co_filename, obj.f_lineno)
        else:
            S = str(obj)

        if obj is pop:
            stack.pop()
            # break
            continue

        print('-' * len(stack), S, end='')

        if I in stack:
            print(' Recurse')
            continue
        elif I in visited:
            print(' Visited')
            continue
        elif len(stack) >= maxdepth:
            print(' Depth limit')
            continue
        else:
            print(' ->')

        stack.append(I)
        visited.add(I)

        todo.insert(0, pop)

        for R in gc.get_referrers(obj):
            if R is top or R is next or R is todo:
                continue
            todo.insert(0, R)

class RegularNamedTemporaryFile(object):
    """Like tempfile.NamedTemporaryFile which doesn't use O_TEMPORARY on windows
    """
    def __init__(self, *args, **kws):
        mkstemp_kws = {k: v for k, v in kws.items()
                       if k in ['suffix', 'prefix', 'dir']}
        fd, self.name = tempfile.mkstemp(**mkstemp_kws)
        try:
            if 'mode' in kws:
                self.file = os.fdopen(fd, kws['mode'])
            else:
                self.file = os.fdopen(fd)
            self.read = self.file.read
            self.write = self.file.write
            self.flush = self.file.flush
            self.seek = self.file.seek
        except:
            os.unlink(self.name)
            raise

    def __del__(self):
        self.close()

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
            os.unlink(self.name)
            self.file = None
            self.read = self.write = self.flush = self.seek = None
