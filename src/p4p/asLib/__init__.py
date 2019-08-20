import logging
import warnings
import socket
import re

from threading import Lock
from collections import defaultdict
from functools import partial
from weakref import WeakKeyDictionary

from .yacc import parse, ACFError

from ..client.thread import Context, LazyRepr

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
    Context = Context # allow tests to replace

    defaultACF = """
    ASG(DEFAULT) {
        RULE(1, WRITE)
        RULE(1, UNCACHED)
    }
    """
    def __init__(self, acf = None):
        self._lock = Lock()
        # {Channel:(group, user, host, level)}
        self._anodes = WeakKeyDictionary()
        self._ctxt = None
        self._inputs = {}
        self._subscriptions = {}

        self.parse(acf or self.defaultACF)

    def parse(self, acf):
        ast = parse(acf)

        # map user or host to set of groups
        uag = defaultdict(set)
        hag = defaultdict(set)
        uags, hags = set(), set()

        asg = {}
        invars = {}
        inputs = {}
        # mapping from variable name to list of ASGs which reference it
        vargroups = defaultdict(set)

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

                rules = asg[node[1]] = []
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
                                    vargroups[var].add(node[1])
                                    inputs[var] = None

                                # cheating here by using python expression syntax instead of CALC.
                                try:
                                    expr = compile(rnode[1], '<acf>', 'single')
                                except SyntaxError:
                                    _log.exception("Error in CALC expression")
                                    # default to false on error
                                    expr = compile('0', '<acf>', 'single')

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
                        invars[anode[1]] = anode[2]

                    else:
                        warnings.warn("Invalid Rule AST: %s"%(anode,))

            else:
                warnings.warn("Invalid AST: %s"%(node,))

        hag_addr = self._resolve_hag(hag)

        # prevent accidental insertions
        uag = dict(uag)
        hag = dict(hag)
        vargroups = dict(vargroups)

        # at this point, success is assumed.
        # aka. errors will not be clean

        if invars and self._ctxt is None:
            self._ctxt = self.Context('pva')

        # cancel any active subscriptions
        [S.close() for S in self._subscriptions.values()]

        with self._lock:
            self._uag = uag
            self._hag = hag
            self._asg = asg
            self._asg_DEFAULT = asg.get('DEFAULT', [])
            self._hag_addr = hag_addr
            self._inputs = inputs
            self._vargroups = vargroups

        self._recompute()

        # create new subscriptions
        # which will trigger a lot of recomputes
        self._subscriptions = {var: self._ctxt.monitor(pv, partial(self._var_update, var), notify_disconnect=True) for var,pv in invars.items()}

    def _var_update(self, var, value):
        # clear old value first
        val = None
        try:
            if value is not None:
                val = float(value.value)
        except:
            _log.exception('INP%s unable to store %s', var, LazyRepr(value))

        with self._lock:
            self._inputs[var] = val
            asgs = self._vargroups.get(var)

        if asgs:
            self._recompute(only=asgs)

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
        _hag_addr = _resolve_hag(self._hag)
        with self._lock:
            self._hag_addr = _hag_addr

        self._recompute()

    def create(self, channel, group, user, host, level, roles=[]):
        # Default to restrictive.  Used in case of error
        perm = 0

        with self._lock:

            uags = self._uag.get(user, set())
            for role in roles:
                uags |= self._uag.get('role:'+role, set())
            hags = self._hag_addr.get(host, set())
            rules = self._asg.get(group, self._asg_DEFAULT)

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
                                accept = float(eval(cond[2], {}, self._inputs) or 0.0) >= 0.5 # horray for legacy... I mean compatibility
                            except:
                                # this could be any of a number of exceptions
                                # which all add up to the same.  Invalid expression
                                accept = False
                                _log.exception('Error evaluating: %s with %s', cond[1], self._inputs)
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
