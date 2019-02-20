"""
See https://github.com/epics-extensions/ca-gateway/blob/master/docs/GATEWAY.pvlist
"""

import re
from collections import defaultdict, OrderedDict

def _re_join(exprs, capture=''):
    A = '|'.join(['(%s%s)'%(capture, E) for E in exprs])
    return re.compile('^(?:%s)$'%A)

class PVList(object):
    def __init__(self, pvl):
        allowfirst = False

        # {'host':[RE, ...]}
        deny_from = defaultdict(list)
        deny_all = set()
        allow = OrderedDict()

        lineno = 0
        for line in (pvl or '.* ALLOW').splitlines():
            lineno += 1
            try:
                line = line.strip()
                if len(line)==0 or line[:1]=='#':
                    continue

                M = re.match(r'^\s*EVALUATION\s+ORDER\s+([A-Z]+),\s+([A-Z]+)\s*$', line)
                if M:
                    if M.groups()==('DENY', 'ALLOW'):
                        allowfirst = True # allow rules take precedence
                    elif M.groups()==('ALLOW', 'DENY'):
                        allowfirst = False # deny rules take precedence (default)
                    else:
                        raise RuntimeError("Invalid order: %s"%(M.groups(),))
                    continue

                parts = [part.strip() for part in line.split(None)]
                pattern, cmd, parts = parts[0], parts[1], parts[2:]

                # test compile
                C = re.compile(pattern)
                if C.groups>0:
                    raise RuntimeError("Capture groups and ALIAS not yet supported")

                if cmd=='DENY':
                    if len(parts) and parts[0]=='FROM':
                        parts = parts[1:]

                    if parts:
                        for host in parts:
                            deny_from[host].append(pattern)

                    else:
                        deny_all.add(pattern)

                elif cmd=='ALIAS':
                    raise RuntimeError("ALIAS not yet supported")

                elif cmd=='ALLOW':
                    asg = parts[0] if len(parts)>0 else 'DEFAULT'
                    asl = int(parts[1] if len(parts)>1 else '0')

                    allow[pattern] = (None, asg, asl)

                else:
                    raise RuntimeError("Unknown command: %s"%cmd)

            except Exception as e:
                raise
                #raise e.__class__("Error on line %s: %s"%(lineno, e))

        deny_all = list(deny_all)

        # RE's for each host specific list also include the general list.
        # So only need to run one deny RE for request
        self._deny_from = {addr:_re_join(exprs+deny_all, '?:') for addr, exprs in deny_from.items()}
        self._deny_all = _re_join(deny_all, '?:')

        allow_pat, self._allow_actions = list(allow.keys()), list(allow.values())
        # ALLOW entries are given in order of increascing precedence.
        # The last match in the file is used.
        allow_pat.reverse()
        self._allow_actions.reverse()

        self._allow_pat = _re_join(allow_pat)
        assert self._allow_pat.groups==len(allow)

        self._allow_groups = tuple(range(1, 1+len(allow)))

    def compute(self, pv, addr):
        P = self._deny_from.get(addr, self._deny_all)

        if not P.match(pv):
            M = self._allow_pat.match(pv)
            if M:
                for idx, val in enumerate(M.group(*self._allow_groups)):
                    if val is not None:
                        _alias, asg, asl = self._allow_actions[idx]

                        return pv, asg, asl

        return None, None, None
