import logging
import warnings
import socket
import re

from threading import Lock
from collections import defaultdict
from functools import partial
from weakref import WeakKeyDictionary

from .yacc import parse, ACFError

from .. import Value
from ..client.thread import Context, LazyRepr, Disconnected

_log = logging.getLogger(__name__)

READ = 1
PUT = 2
RPC = 4
UNCACHED = 8
WRITE = READ | PUT | RPC
actionmask = {
    'READ':READ,
    'WRITE':WRITE,
    'PUT':PUT,
    'RPC':RPC,
    'UNCACHED':UNCACHED,
}

class Engine(object):
    '''Access Security File (ACF) parsing and evaluation engine

    >>> with open(fname, 'r') as F:
            acf = Engine(F.read())
    '''

    defaultACF = """
    ASG(DEFAULT) {
        RULE(1, WRITE)
        RULE(1, UNCACHED)
    }
    """
    def __init__(self, acf = None, ctxt = None):
        self._lock = Lock()
        # {Channel:(group, user, host, level)}
        self._anodes = WeakKeyDictionary()
        self._ctxt = ctxt
        self._asg = {}
        self._subscriptions = {}

        self.parse(acf or self.defaultACF)

    def report(self):
        A, B, C, D = [], [], [], []
        for asg,(rules,inputs) in self._asg.items():
            for var, val in inputs.items():
                A.append(asg)
                B.append(var)
                C.append(val or 0.0)
                D.append(val is not None)
        return {
            'value.asg':A,
            'value.var':B,
            'value.value':C,
            'value.connected':D,
        }

    def parse(self, acf):
        ast = parse(acf)

        # map user or host to set of groups
        uag = defaultdict(set)
        hag = defaultdict(set)
        uags, hags = set(), set()

        # {'name';([rules], {vars})}
        asg = {}
        # {'pvname':[('name','VAR')]}
        invars = defaultdict(list)

        for node in ast:
            if node[0]=='UAG':
                # ('UAG', name, [members...])
                uags.add(node[1])
                for member in node[2]:
                    uag[member].add(node[1])

            elif node[0]=='HAG':
                # ('HAG', name, [members...])
                hags.add(node[1])
                for member in node[2]:
                    hag[member].add(node[1])

            elif node[0]=='ASG':
                # ('ASG', name, [rules...])
                #   rule : ('INP', 'A', 'pv:name')
                #        | ('RULE', 1, 'WRITE', trap, None | [])

                rules, inputs = asg[node[1]] = [], {}
                for anode in node[2]:
                    if anode[0]=='RULE':
                        rule = []
                        for rnode in anode[4] or []:
                            if rnode[0] in ('UAG', 'HAG'):
                                # ('UAG', ['name'])
                                # ('HAG', ['name'])
                                rule.append((rnode[0], set(rnode[1])))

                            elif rnode[0]=='CALC':
                                # ('CALC', '<expr>')
                                for var in re.findall(r'[A-Z]', rnode[1]):
                                    inputs[var] = None

                                # cheating here by using python expression syntax instead of CALC.
                                try:
                                    expr = compile(rnode[1], '<acf>', 'eval')
                                    _log.debug('Compile %s', rnode[1])
                                except SyntaxError:
                                    _log.exception("Error in CALC expression")
                                    # default to false on error
                                    expr = compile('0', '<acf>', 'eval')

                                rule.append((rnode[0], rnode[1], expr))

                            else:
                                warnings.warn("Invalid RULE condition AST: %s"%(rnode,))

                        try:
                            mask = actionmask[anode[2]]
                        except KeyError:
                            _log.warn('Ignoring unknown permission "%s"', anode[2])
                            mask = 0 # grant no permissions
                        rules.append( (mask, anode[1], anode[3], rule) )

                    elif anode[0]=='INP':
                        # ('INP', 'A', 'pv:name')
                        invars[anode[2]].append((node[1], anode[1]))

                    else:
                        warnings.warn("Invalid Rule AST: %s"%(anode,))

            else:
                warnings.warn("Invalid AST: %s"%(node,))

        hag_addr = self._resolve_hag(hag)

        # prevent accidental insertions
        uag = dict(uag)
        hag = dict(hag)
        invars = dict(invars)

        # at this point, success is assumed.
        # aka. errors will not be clean

        if invars and self._ctxt is None:
            for pv, grps in invars.items():
                _log.warning('No Client to connect to ACF %s for %s', pv, grps)

        # cancel any active subscriptions
        [S.close() for S in self._subscriptions.values()]

        with self._lock:
            self._uag = uag
            self._hag = hag
            self._asg = asg
            self._asg_DEFAULT = asg.get('DEFAULT', [])
            self._hag_addr = hag_addr

        self._recompute()

        # create new subscriptions
        # which will trigger a lot of recomputes
        if self._ctxt is not None:
            [_log.debug('subscribing to %s for %s', pv, grps) for pv,grps in invars.items()]
            self._subscriptions = {pv: self._ctxt.monitor(pv, partial(self._var_update, grps), notify_disconnect=True) for pv,grps in invars.items()}
        else:
            self._subscriptions = {}

    def _var_update(self, grps, value):
        # clear old value first
        val = None
        if not isinstance(value, Disconnected):
            try:
                val = float(value or 0.0)
            except:
                _log.exception('INP%s unable to store %s', grps, LazyRepr(value))

        _log.debug('Update INP%s <- %s', grps, val)

        with self._lock:
            for asg, var in grps:
                _rules, inputs = self._asg[asg]
                inputs[var] = val

        if grps:
            self._recompute(only={asg for asg,var in grps})

    def _recompute(self, only=None):
        _log.debug("Recompute %s", only or "all")
        anodes, self._anodes = self._anodes, WeakKeyDictionary()

        for channel, (group, user, host, level) in anodes.items():
            if only is None or group in only:
                self.create(channel, group, user, host, level)
            else:
                self._anodes[channel] = (group, user, host, level)

    @staticmethod
    def _gethostbyname(host):
        try:
            return socket.gethostbyname(host)
        except socket.gaierror as e:
            _log.warn( "Ignore invalid hostname \"%s\" : %s", host, e )

    def _resolve_hag(self, _hag):
        hag_addr = defaultdict(set)

        for host, groups in _hag.items():
            ip = self._gethostbyname(host)
            if ip is not None:
                hag_addr[ip] |= groups

        return dict(hag_addr)

    def resolve_hag(self):
        # TODO: racy.  How to make atomic w/o waiting for DNS lookup with lock?
        _hag_addr = self._resolve_hag(self._hag)
        with self._lock:
            self._hag_addr = _hag_addr

        self._recompute()

    def create(self, channel, group, user, host, level, roles=[]):
        # Default to restrictive.  Used in case of error
        perm = 0
        _log.debug('(re)create %s, %s, %s, %s, %s', channel, group, user, host, level)

        with self._lock:

            uags = self._uag.get(user, set())
            for role in roles:
                uags |= self._uag.get('role/'+role, set())
            hags = self._hag_addr.get(host, set())
            rules, inputs = self._asg.get(group, self._asg_DEFAULT)

            trapit = False
            try:
                for mask, asl, trap, conds in rules:
                    accept = True
                    for cond in conds:
                        if cond[0]=='UAG':
                            accept = len(cond[1].intersection(uags))
                        elif cond[0]=='HAG':
                            accept = len(cond[1].intersection(hags))
                        elif cond[0]=='CALC':
                            try:
                                accept = float(eval(cond[2], {}, inputs) or 0.0) >= 0.5 # horray for legacy... I mean compatibility
                            except:
                                # this could be any of a number of exceptions
                                # which all add up to the same.  Invalid expression
                                accept = False
                                _log.exception('Error evaluating: %s with %s', cond[1], [(k,v,type(v)) for k,v in inputs.items()])
                            else:
                                _log.debug('Evaluate %s with %s -> %s', cond, [(k,v,type(v)) for k,v in inputs.items()], accept)
                        else:
                            warnings.warn("Invalid AST RULE: %s"%cond)
                            accept = False

                        if not accept:
                            break

                    if accept:
                        trapit |= trap
                        perm |= mask

            except:
                _log.exception("Error while calculating ASG for %s, %s, %s, %s, %s",
                            channel, group, user, host, level)

            put = perm & PUT
            rpc = perm & RPC
            uncached = perm & UNCACHED

            channel.access(put=bool(put), rpc=bool(rpc), uncached=bool(uncached), audit=trapit)

            self._anodes[channel] = (group, user, host, level)

    def _check_host(self, hag, user, host):
        groups = self._hag_addr.get(host) or set()
        return hag in groups

    def _check_user(self, uag, user, host):
        groups = self._uag.get(user) or set()
        return uag in groups
