"""
See https://github.com/epics-extensions/ca-gateway/blob/master/docs/GATEWAY.pvlist
"""

import socket
import logging
import re
from collections import defaultdict, OrderedDict

_log = logging.getLogger(__name__)

def _sub_add(expr, ngroups, adjust=0):
    '''Adjust RE substitution offsets
    '''
    def check(M):
        grp = int(M.group(1))
        if grp<1 or grp>ngroups:
            raise ValueError('Out of range substitution %d.  Must be in [1, %d]'%(grp,ngroups))
        return r'\%d'%(grp+adjust)
    return re.sub(r'(?<!\\)\\([0-9]+)', check, expr)

def _re_join(exprs, capture=''):
    A = '|'.join(['(%s%s)'%(capture, E) for E in exprs])
    return re.compile('^(?:%s)$'%A)

class PVList(object):
    '''Parse and prepare pvlist text.

    >>> with open(fname, 'r') as F:
            pvl = PVList(F.read())
    '''
    def __init__(self, pvl):
        allowfirst = False

        # {'host':[RE, ...]}
        deny_from = defaultdict(set)
        deny_all = set()
        allow = OrderedDict() # {RE:(sub|None, asg, asl)}
        # number of match groups encountered so far.
        # must match between allow key (pattern) and substitution
        ngroups = 1 # one indexed
        self._allow_groups = []

        lines = (pvl or '.* ALLOW').splitlines()
        # ALLOW entries are given in order of increasing precedence.
        # The last match in the file is used.
        lines.reverse()

        lineno = 0
        for line in lines:
            lineno += 1
            try:
                line = line.strip()
                if len(line)==0 or line[:1]=='#':
                    continue

                M = re.match(r'^\s*EVALUATION\s+ORDER\s+([A-Z]+),\s+([A-Z]+)\s*$', line)
                if M:
                    if M.groups()==('DENY', 'ALLOW'):
                        allowfirst = True # allow rules take precedence
                        _log.warn('Ignoring "EVALUATION ORDER DENY, ALLOW".  Only ALLOW, DENY is implemented.')
                    elif M.groups()==('ALLOW', 'DENY'):
                        allowfirst = False # deny rules take precedence (default)
                    else:
                        raise RuntimeError("Invalid order: %s"%(M.groups(),))
                    continue

                parts = [part.strip() for part in line.split(None)]
                pattern, cmd, parts = parts[0], parts[1].upper(), parts[2:]

                # test compile
                C = re.compile(pattern)

                if pattern in allow:
                    continue # ignore duplicate pattern

                if cmd=='DENY':
                    if len(parts) and parts[0].upper()=='FROM':
                        parts = parts[1:]

                    if parts:
                        for host in parts:
                            host = self._gethostbyname(host)
                            deny_from[host].add(pattern)

                    else:
                        deny_all.add(pattern)

                elif cmd=='ALIAS':
                    self._allow_groups.append(ngroups)
                    ngroups += 1 # _re_join adds one capture group

                    alias = _sub_add(parts[0], ngroups=C.groups, adjust=ngroups-1)
                    asg = parts[1] if len(parts)>1 else 'DEFAULT'
                    asl = int(parts[2] if len(parts)>2 else '0')

                    allow[pattern] = (alias, asg, asl)
                    ngroups += C.groups

                elif cmd=='ALLOW':
                    self._allow_groups.append(ngroups)
                    ngroups += 1 # _re_join adds one capture group

                    asg = parts[0] if len(parts)>0 else 'DEFAULT'
                    asl = int(parts[1] if len(parts)>1 else '0')

                    allow[pattern] = (None, asg, asl)

                else:
                    raise RuntimeError("Unknown command: %s"%cmd)

            except Exception as e:
                raise e.__class__("Error on line %s: %s"%(lineno, e))

        deny_all = list(deny_all)

        # RE's for each host specific list also include the general list.
        # So only need to run one deny RE for request
        self._deny_from = {addr:_re_join(list(exprs)+deny_all, '?:') for addr, exprs in deny_from.items()}
        self._deny_all = _re_join(deny_all, '?:')

        allow_pat, self._allow_actions = list(allow.keys()), list(allow.values())

        self._allow_pat = _re_join(allow_pat)

        assert self._allow_pat.groups+1==ngroups, (self._allow_pat.groups, ngroups)

    @staticmethod
    def _gethostbyname(host):
        return socket.gethostbyname(host)

    def compute(self, pv, addr):
        '''Lookup PV name and client IP address.

        Returns a triple of None/rewritten PV name, security group name (ASG), and security level number (ASL).
        '''
        pv = pv.decode('UTF-8')
        P = self._deny_from.get(addr, self._deny_all)

        if not P.match(pv):
            M = self._allow_pat.match(pv)
            if M:
                for idx, val in enumerate(M.group(*self._allow_groups)):
                    if val is not None:
                        alias, asg, asl = self._allow_actions[idx]
                        if alias is not None:
                            pv = M.expand(alias)

                        return pv, asg, asl

        return None, None, None
