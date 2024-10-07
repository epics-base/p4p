# exec()d by setup.py and gha-set-pre.py

class Version(object):
    """P4P Version number
    
    Comparible with version string, or a tuple of integers with len() <= 5.
    """
    def __init__(self, s):
        self._parts = self.__parse(s)

    @staticmethod
    def __parse(s):
        import re
        M = re.match(r'^(\d+)\.(\d+)\.(\d+)(?:([ab])(\d+))?$', s)
        if M is None:
            raise ValueError('Invalid version %r'%s)
        x,y,z,abr,n = M.groups()
        abr = {
            None:0,
            'a':-2,
            'b':-1,
        }[abr]
        assert abr is not None or n=='0', s
        return (int(x),int(y),int(z), abr, int(n or '0'))

    @property
    def is_release(self):
        return self._parts[3]==0

    def __repr__(self):
        s = str(self)
        return 'Version(%r)'%s

    def __str__(self):
        x,y,z,abr,n = self._parts
        ver = '%d.%d.%d'%(x,y,z)
        abr = {0:'', -1:'b', -2:'a'}[abr]
        if abr:
            ver = ver + '%s%d'%(abr, n)
        return ver

    def _cmp(self, o):
        if isinstance(o, str):
            o = self.__parse(o)
        o = tuple(o)
        if len(o) < 5:
            o = o + (0,)*(5 - len(o))

        elif len(o) > 5:
            raise ValueError("must compare with iterable len() <= 5")

        for l,r in zip(self._parts, o):
            if l < r:
                return -1
            elif l > r:
                return 1
        return 0

    def __lt__(self, o):
        return self._cmp(o)<0
    def __le__(self, o):
        return self._cmp(o)<=0
    def __eq__(self, o):
        return self._cmp(o)==0
    def __ge__(self, o):
        return self._cmp(o)>=0
    def __gt__(self, o):
        return self._cmp(o)>0

version = Version('4.2.0')
