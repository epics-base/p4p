"""Python reference counter statistics.
"""

from __future__ import print_function

import sys
import gc
import inspect
import time
from glob import fnmatch
try:
    from types import InstanceType
except ImportError:  # py3
    InstanceType = None


class StatsDelta(object):

    """GC statistics tracking.

    Monitors the number of instances of each type/class (cf. gcstats())
    and prints a report of any changes (ie. new types/count changes).
    Intended to assist in detecting reference leaks.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset internal statistics counters
        """
        self.stats, self.ntypes = None, None

    def collect(self, file=sys.stderr):
        """Collect stats and print results to file

        :param file: A writable file-like object
        """
        cur = gcstats()
        Ncur = len(cur)
        if self.stats is not None and file is not None:
            prev = self.stats
            Nprev = self.ntypes  # may be less than len(prev)

            if Ncur != Nprev:
                print("# Types %d -> %d" % (Nprev, Ncur), file=file)

            Scur, Sprev, first = set(cur), set(prev), True
            for T in Scur - Sprev:  # new types
                if first:
                    print('New Types', file=file)
                    first = False
                print(' ', T, cur[T], file=file)

            first = True
            for T in Sprev - Scur:  # collected types
                if first:
                    print('Cleaned Types', file=file)
                    first = False
                print(' ', T, -prev[T], file=file)

            first = True
            for T in Scur & Sprev:
                if cur[T] == prev[T]:
                    continue
                if first:
                    print('Known Types', file=file)
                    first = False
                print(' ', T, cur[T], 'delta', cur[T] - prev[T], file=file)

        else:  # first call
            print("All Types", file=file)
            for T, C in cur.items():
                print(' ', T, C, file=file)

        self.stats, self.ntypes = cur, len(cur)
        # gc.collect()


def gcstats():
    """Count the number of instances of each type/class

    :returns: A dict() mapping type (as a string) to an integer number of references
    """
    all = gc.get_objects()
    _stats = {}

    for obj in all:
        K = type(obj)
        if K is StatsDelta:
            continue  # avoid counting ourselves

        elif K is InstanceType:  # instance of an old-style class
            K = getattr(obj, '__class__', K)

        # Track types as strings to avoid holding references
        K = str(K)

        try:
            _stats[K] += 1
        except KeyError:
            _stats[K] = 1

    # explicitly break the reference loop between the list and this frame,
    # which is contained in the list
    # This would otherwise prevent the list from being free'd
    del all

    return _stats


def gcfind(name):
    all = gc.get_objects()
    found = []

    for obj in all:
        K = type(obj)
        if K is gcfind:
            continue  # avoid counting ourselves

        if K is InstanceType:  # instance of an old-style class
            K = getattr(obj, '__class__', K)

        if fnmatch(str(K), name):
            found.append(obj)

    return found


class _StatsThread(object):

    def __init__(self, period, file):
        self.period, self.file = period, file
        self.S = StatsDelta()

    def __call__(self):
        while True:
            self.S.collect(file=self.file)
            time.sleep(self.period)


def periodic(period=60.0, file=sys.stderr):
    """Start a daemon thread which will periodically print GC stats

    :param period: Update period in seconds
    :param file: A writable file-like object
    """
    import threading
    import time
    S = _StatsThread(period=period, file=file)
    T = threading.Thread(target=S)
    T.daemon = True
    T.start()

if __name__ == '__main__':
    # for T,C in gcstats().items():
    #  print T,C
    gc.set_debug(gc.DEBUG_COLLECTABLE)
    S = StatsDelta()
    while True:
        S.collect()
        # gc.collect()
