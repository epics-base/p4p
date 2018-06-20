
from __future__ import print_function

import gc, inspect, unittest
import fnmatch

from .. import listRefs

class RefTestMixin(object):
    # set to list of names to compare.  Set to None to disable
    ref_check = ('*',)
    def setUp(self):
        if self.ref_check is not None:
            refs = self.__raw_before = listRefs()
            self.__names = set()
            for pat in self.ref_check:
                self.__names |= set(fnmatch.filter(refs, pat))
            self.__before = dict([(K,V) for K,V in refs.items() if K in self.__names])

        super(RefTestMixin, self).setUp()
    def tearDown(self):
        super(RefTestMixin, self).tearDown()
        if self.ref_check is not None:
            gc.collect()
            refs = listRefs()
            after = dict([(K,V) for K,V in refs.items() if K in self.__names])
            # the compared ref counters should be unchanged after each case
            self.assertDictEqual(self.__before, after)
            # check for any obviously corrupt counters, even those not being compared
            self.assertFalse(any([V>1000000 for V in refs.values()]), "before %s after %s"%(self.__raw_before, refs))

class RefTestCase(RefTestMixin, unittest.TestCase):
    def setUp(self):
        super(RefTestCase, self).setUp()
    def tearDown(self):
        super(RefTestCase, self).tearDown()

def gctrace(obj, maxdepth=8):
    # depth first traversal
    pop = object()
    top = inspect.currentframe()
    next = top.f_back
    stack, todo = [], [obj]
    visited = set()

    while len(todo):
        obj = todo.pop(0)
        #print('N', obj)
        I = id(obj)
        if inspect.isframe(obj):
            S = 'Frame %s:%d'%(obj.f_code.co_filename, obj.f_lineno)
        else:
            S = str(obj)

        if obj is pop:
            stack.pop()
            #break
            continue

        print('-'*len(stack), S, end='')

        if I in stack:
            print(' Recurse')
            continue
        elif I in visited:
            print(' Visited')
            continue
        elif len(stack)>=maxdepth:
            print(' Depth limit')
            continue
        else:
            print(' ->')

        stack.append(I)
        visited.add(I)

        todo.insert(0,pop)

        for R in gc.get_referrers(obj):
            if R is top or R is next or R is todo:
                continue
            todo.insert(0,R)
