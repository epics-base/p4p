
from __future__ import print_function

import gc, inspect

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
