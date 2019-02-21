import logging
import warnings
import socket

from threading import Lock
from collections import defaultdict
from functools import partial
from weakref import WeakKeyDictionary

from .yacc import parse

_log = logging.getLogger(__name__)

READ = 1
WRITEONLY = 2
WRITE = 3 # implies READ
RPC = 4
UNCACHED = 8
actionmask = {
    'READ':READ,
    'WRITE':WRITE,
    'RPC':RPC,
    'UNCACHED':UNCACHED,
}

class Engine(object):
    defaultACF = """
    ASG(DEFAULT) {
        RULE(1, WRITE)
        RULE(1, RPC)
        RULE(1, UNCACHED)
    }
    """
    def __init__(self, acf = None):
        self._lock = Lock()
        self._anodes = WeakKeyDictionary()
        self.parse(acf or self.defaultACF)

    def parse(self, acf):
        ast = parse(acf)

        # map user or host to set of groups
        _uag = defaultdict(set)
        _hag = defaultdict(set)
        uags, hags = set(), set()

        _asg = {}

        for node in ast:
            if node[0]=='UAG':
                # ('UAG', name, [members...])
                uags.add(node[1])
                for member in node[2]:
                    _uag[member].add(node[1])

            elif node[0]=='HAG':
                # ('HAG', name, [members...])
                hags.add(node[1])
                for member in node[2]:
                    _hag[member].add(node[1])

            elif node[0]=='ASG':
                # ('ASG', name, [rules...])
                #   rule : ('INP', 'A', 'pv:name')
                #        | ('RULE', 1, 'WRITE', trap, None | [])

                # re-write only 'RULE' as (actionmask, asl, trap, conditions)
                _asg[node[1]] = [(actionmask.get(rule[2],0), rule[1], rule[3], rule[4] or []) for rule in node[2] if rule[0]=='RULE']

            else:
                warnings.warn("Invalid AST: %s"%node)

        _hag_addr = self._resolve_hag(_hag)

        # prevent accidental insertions
        _uag = dict(_uag)
        _hag = dict(_hag)

        with self._lock:
            self._uag = _uag
            self._hag = _hag
            self._asg = _asg
            self._asg_DEFAULT = _asg.get('DEFAULT', [])
            self._hag_addr = _hag_addr

        self._recompute()

    def _recompute(self):
        anodes, self._anodes = self._anodes, WeakKeyDictionary()

        for channel, args in anodes.items():
            self.create(channel, *args)

    @staticmethod
    def _gethostbyname(host):
        return socket.gethostbyname(host)

    def _resolve_hag(self, _hag):
        hag_addr = defaultdict(set)

        for host, groups in _hag.items():
            ip = self._gethostbyname(host)
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

        uags = self._uag.get(user, set())
        for role in roles:
            uags |= self._uag.get('role:'+role, set())
        hags = self._hag_addr.get(host, set())
        rules = self._asg.get(group, self._asg_DEFAULT)

        try:
            for mask, asl, trap, conds in rules:
                accept = True
                for cond in conds:
                    if cond[0]=='UAG':
                        accept = cond[1] in uags
                    elif cond[0]=='HAG':
                        accept = cond[1] in hags
                    elif cond[0]=='CALC':
                        accept = False # TODO
                    else:
                        warnings.warn("Invalid AST RULE: %s"%cond)
                        accept = False
                    if not accept:
                        break

                if accept:
                    perm |= mask

        except:
            _log.exception("Error while calculating ASG for %s, %s, %s, %s, %s",
                           channel, group, user, host, level)

        put = perm & WRITEONLY
        rpc = perm & RPC
        uncached = perm & UNCACHED

        channel.access(put=bool(put), rpc=bool(rpc), uncached=bool(uncached))

        self._anodes[channel] = (group, user, host, level)

    def _check_host(self, hag, user, host):
        groups = self._hag_addr.get(host) or set()
        return hag in groups

    def _check_user(self, uag, user, host):
        groups = self._uag.get(user) or set()
        return uag in groups
