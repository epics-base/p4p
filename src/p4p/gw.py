from __future__ import print_function

import sys
import os
import logging
import time
import socket
import threading
import platform
import pprint
import json
import re
import sqlite3

from functools import wraps, reduce

from .nt import NTScalar, Type, NTTable
from .server import Server, StaticProvider, removeProvider
from .server.thread import SharedPV, RemoteError
from .client.thread import Context
from . import set_debug, listRefs
from . import _gw
from .asLib import Engine, ACFError
from .asLib.pvlist import PVList
from .test.utils import RegularNamedTemporaryFile as NamedTemporaryFile

if sys.version_info >= (3, 0):
    unicode = str

_log = logging.getLogger(__name__)
_log_audit = logging.getLogger(__name__+'.audit')

def uricall(fn):
    @wraps(fn)
    def rpc(self, pv, op):
        ret = fn(self, op, **dict(op.value().query.items()))
        op.done(ret)
    return rpc

class TestChannel(object):
    def __init__(self):
        self.perm = None
    def access(self, **kws):
        self.perm = kws

class TableBuilder(object):
    def __init__(self, colinfo):
        self.labels, cols = [], []
        for type, name, label in colinfo:
            self.labels.append(label)
            cols.append((name, 'a'+type))

        self.type = NTTable.buildType(cols)

    def wrap(self, values):
        S, NS = divmod(time.time(), 1.0)
        ret = self.type({
            'labels':self.labels,
            'timeStamp': {
                'secondsPastEpoch': S,
                'nanoseconds': NS * 1e9,
            },
        })
        # unzip list of tuple into tuple of lists
        cols = list([] for k in ret.value)

        for row in values:
            for I, C in zip(row, cols):
                C.append(I)

        for k, C in zip(ret.value, cols):
            ret.value[k] = C

        return ret

    def unwrap(self, value):
        return value

class RefAdapter(object):
    def __init__(self):
        self.type = NTTable.buildType([('type', 'as'), ('count', 'aI'), ('delta', 'ai')])
        self.prev = {}
        self._labels = ['Type', 'Count', 'Delta']

    def wrap(self, cnts):
        kcur = set(cnts)
        kprev = set(self.prev)

        update = []

        added = kcur - kprev
        for k in added:
            update.append((k, cnts[k], cnts[k]))

        removed = kprev - kcur
        for k in removed:
            update.append((k, 0, -self.prev[k]))

        for k in kcur&kprev:
            c, p = cnts[k], self.prev[k]
            if c!=p:
                update.append((k, c, c-p))

        update.sort(key=lambda t:t[0])
        self.prev = cnts

        Ns, Cs, Ds = [], [], []
        for N, C, D in update:
            Ns.append(N)
            Cs.append(C)
            Ds.append(D)

        V = self.type({
            'value.type':Ns,
            'value.count':Cs,
            'value.delta':Ds,
            'timeStamp.secondsPastEpoch':time.time(),
        })
        if self._labels is not None:
            self._labels, V.labels = None, self._labels

        return V

    def unwrap(self, V):
        return V

statsType = Type([
    ('ccacheSize', NTScalar.buildType('L')),
    ('mcacheSize', NTScalar.buildType('L')),
    ('gcacheSize', NTScalar.buildType('L')),
    ('banHostSize', NTScalar.buildType('L')),
    ('banPVSize', NTScalar.buildType('L')),
    ('banHostPVSize', NTScalar.buildType('L')),
], id='epics:p2p/Stats:1.0')

permissionsType = Type([
    ('pv', 's'),
    ('account', 's'),
    ('peer', 's'),
    ('roles', 'as'),
    ('asg', 's'),
    ('asl', 'i'),
    ('permission', ('S', None, [
        ('put', '?'),
        ('rpc', '?'),
        ('uncached', '?'),
        ('audit', '?'),
    ])),
], id='epics:p2p/Permission:1.0')

asDebugType = NTTable.buildType([
    ('asg', 'as'),
    ('var', 'as'),
    ('value', 'ad'),
    ('connected', 'a?'),
])

class GWStats(object):
    """I manage statistics for all GWHandler instances
    """
    def __init__(self, statsdb=None):
        self.statsdb = statsdb

        self.handlers = [] # GWHandler instances we derive stats from

        self._pvs = {} # name suffix -> SharedPV

        if not statsdb:
            self.__tempdb = NamedTemporaryFile()
            self.statsdb = self.__tempdb.name
            _log.debug("Using temporary stats db: %s", self.statsdb)

        with sqlite3.connect(self.statsdb) as C:
            C.executescript("""
                DROP TABLE IF EXISTS us;
                DROP TABLE IF EXISTS usbyname;
                DROP TABLE IF EXISTS usbypeer;
                DROP TABLE IF EXISTS ds;
                DROP TABLE IF EXISTS dsbyname;
                DROP TABLE IF EXISTS dsbypeer;

                CREATE TABLE us(
                    usname REAL NOT NULL,
                    optx INTEGER NOT NULL,
                    oprx REAL NOT NULL,
                    peer STRING NOT NULL,
                    trtx REAL NOT NULL,
                    trrx REAL NOT NULL
                );
                CREATE TABLE ds(
                    usname STRING NOT NULL,
                    dsname STRING NOT NULL,
                    optx REAL NOT NULL,
                    oprx REAL NOT NULL,
                    account STRING NOT NULL,
                    peer STRING NOT NULL,
                    trtx REAL NOT NULL,
                    trrx REAL NOT NULL
                );

                CREATE TABLE usbyname(
                    usname STRING NOT NULL,
                    tx REAL NOT NULL,
                    rx REAL NOT NULL
                );
                CREATE TABLE dsbyname(
                    usname STRING NOT NULL,
                    tx REAL NOT NULL,
                    rx REAL NOT NULL
                );

                CREATE TABLE usbypeer(
                    peer STRING NOT NULL,
                    tx REAL NOT NULL,
                    rx REAL NOT NULL
                );
                CREATE TABLE dsbypeer(
                    account STRING NOT NULL,
                    peer STRING NOT NULL,
                    tx REAL NOT NULL,
                    rx REAL NOT NULL
                );
            """)

        self.clientsPV = SharedPV(nt=NTScalar('as'), initial=[])
        self._pvs['clients'] = self.clientsPV

        self.cachePV = SharedPV(nt=NTScalar('as'), initial=[])
        self._pvs['cache'] = self.cachePV

        self.statsPV = SharedPV(initial=statsType())
        self._pvs['stats'] = self.statsPV

        self.pokeStats = SharedPV(nt=NTScalar('i'), initial=0)
        @self.pokeStats.put
        def pokeStats(pv, op):
            self.update_stats()
            op.done()
        self._pvs['poke'] = self.pokeStats

        self.refsPV = SharedPV(nt=RefAdapter(), initial={})
        self._pvs['refs'] = self.refsPV

        self.statsTime = SharedPV(nt=NTScalar('d'), initial=0.0)
        self._pvs['StatsTime'] = self.statsTime

        # PVs for bandwidth usage statistics.
        # 2x tables: us, ds
        # 2x groupings: by PV and by peer
        # 2x directions: Tx and Rx

        def addpv(dir='TX', suffix=''):
            pv = SharedPV(nt=TableBuilder([
                ('s', 'name', 'PV'),
                ('d', 'rate', dir+' (B/s)'),
            ]), initial=[])
            self._pvs[suffix] = pv
            return pv

        self.tbl_usbypvtx = addpv(dir='TX', suffix='us:bypv:tx')
        self.tbl_usbypvrx = addpv(dir='RX', suffix='us:bypv:rx')

        self.tbl_dsbypvtx = addpv(dir='TX', suffix='ds:bypv:tx')
        self.tbl_dsbypvrx = addpv(dir='RX', suffix='ds:bypv:rx')

        def addpv(dir='TX', suffix=''):
            pv = SharedPV(nt=TableBuilder([
                ('s', 'name', 'Server'),
                ('d', 'rate', dir+' (B/s)'),
            ]), initial=[])
            self._pvs[suffix] = pv
            return pv

        self.tbl_usbyhosttx = addpv(dir='TX', suffix='us:byhost:tx')
        self.tbl_usbyhostrx = addpv(dir='RX', suffix='us:byhost:rx')

        def addpv(dir='TX', suffix=''):
            pv = SharedPV(nt=TableBuilder([
                ('s', 'account', 'Account'),
                ('s', 'name', 'Client'),
                ('d', 'rate', dir+' (B/s)'),
            ]), initial=[])
            self._pvs[suffix] = pv
            return pv

        self.tbl_dsbyhosttx = addpv(dir='TX', suffix='ds:byhost:tx')
        self.tbl_dsbyhostrx = addpv(dir='RX', suffix='ds:byhost:rx')

    def bindto(self, provider, prefix):
        'Add myself to a StaticProvider'

        for suffix, pv in self._pvs.items():
            provider.add(prefix+suffix, pv)

    def sweep(self):
        for handler in self.handlers:
            handler.sweep()

    def update_stats(self):
        T0 = time.time()
        self.refsPV.post(listRefs())

        with sqlite3.connect(self.statsdb) as C:
            C.executescript('''
                DELETE FROM us;
                DELETE FROM ds;
                DELETE FROM usbyname;
                DELETE FROM dsbyname;
                DELETE FROM usbypeer;
                DELETE FROM dsbypeer;
            ''')

            for handler in self.handlers:
                usr, dsr, period = handler.provider.report()

                C.executemany('INSERT INTO us VALUES (?,?,?,?,?,?)', usr)
                C.executemany('INSERT INTO ds VALUES (?,?,?,?,?,?,?,?)', dsr)

            C.executescript('''
                INSERT INTO usbyname SELECT usname, sum(optx), sum(oprx) FROM us GROUP BY usname;
                INSERT INTO dsbyname SELECT usname, sum(optx), sum(oprx) FROM ds GROUP BY usname;
                INSERT INTO usbypeer SELECT peer, max(trtx), max(trrx) FROM us GROUP BY peer;
                INSERT INTO dsbypeer SELECT account, peer, max(trtx), max(trrx) FROM ds GROUP BY peer;
            ''')

            #TODO: create some indicies to speed up these queries?

            self.tbl_usbypvtx.post(C.execute('SELECT usname, tx as rate FROM usbyname ORDER BY rate DESC LIMIT 10'))
            self.tbl_usbypvrx.post(C.execute('SELECT usname, rx as rate FROM usbyname ORDER BY rate DESC LIMIT 10'))

            self.tbl_usbyhosttx.post(C.execute('SELECT peer, tx as rate FROM usbypeer ORDER BY rate DESC LIMIT 10'))
            self.tbl_usbyhostrx.post(C.execute('SELECT peer, rx as rate FROM usbypeer ORDER BY rate DESC LIMIT 10'))

            self.tbl_dsbypvtx.post(C.execute('SELECT usname, tx as rate FROM dsbyname ORDER BY rate DESC LIMIT 10'))
            self.tbl_dsbypvrx.post(C.execute('SELECT usname, rx as rate FROM dsbyname ORDER BY rate DESC LIMIT 10'))

            self.tbl_dsbyhosttx.post(C.execute('SELECT account, peer, tx as rate FROM dsbypeer ORDER BY rate DESC LIMIT 10'))
            self.tbl_dsbyhostrx.post(C.execute('SELECT account, peer, rx as rate FROM dsbypeer ORDER BY rate DESC LIMIT 10'))

            self.clientsPV.post([row[0] for row in C.execute('SELECT DISTINCT peer FROM us')])

        statsSum = {'ccacheSize.value':0, 'mcacheSize.value':0, 'gcacheSize.value':0,
                    'banHostSize.value':0, 'banPVSize.value':0, 'banHostPVSize.value':0}
        stats = [handler.provider.stats() for handler in self.handlers]
        for key in statsSum:
            for stat in stats:
                statsSum[key] += stat[key]
        self.statsPV.post(statsType(statsSum))

        self.cachePV.post(reduce(set.__or__, [handler.provider.cachePeek() for handler in self.handlers], set()))

        T1 = time.time()

        self.statsTime.post(T1-T0)

class GWHandler(object):
    def __init__(self, acf, pvlist, readOnly=False):
        self.acf, self.pvlist = acf, pvlist
        self.readOnly = readOnly
        self.channels_lock = threading.Lock()
        self.channels = {}

        self.provider = None
        self.getholdoff = None


    def testChannel(self, pvname, peer):
        _log.debug('%s Searching for %s', peer, pvname)
        usname, _asg, _asl = self.pvlist.compute(pvname, peer.split(':',1)[0])

        if not usname:
            _log.debug("Not allowed: %s by %s", pvname, peer)
            return self.provider.BanHostPV
        else:
            ret = self.provider.testChannel(usname.encode('UTF-8'))
            _log.debug("allowed: %s by %s -> %s", pvname, peer, ret)
            return ret

    def makeChannel(self, op):
        _log.debug("Create %s by %s", op.name, op.peer)
        peer = op.peer.split(':',1)[0]
        usname, asg, asl = self.pvlist.compute(op.name, peer)
        if not usname:
            return None
        chan = op.create(usname.encode('UTF-8'))

        with self.channels_lock:
            try:
                channels = self.channels[op.peer]
            except KeyError:
                self.channels[op.peer] = channels = []
            channels.append(chan)

        try:
            if not self.readOnly: # default is RO
                self.acf.create(chan, asg, op.account, peer, asl, op.roles)
            if self.getholdoff is not None:
                chan.access(holdoff=self.getholdoff)
        except:
            # create() should fail secure.  So allow this client to
            # connect R/O.  We already acknowledged the search, so
            # if we fail here the client will go into a reset loop.
            _log.exception("Default restrictive for %s from %s", op.name, op.peer)
        return chan

    def audit(self, msg):
        _log_audit.info('%s', msg)

    def sweep(self):
        self.provider.sweep()
        replace = {}
        with self.channels_lock:
            for K, chans in self.channels.items():
                chans = [chan for chan in chans if not chan.expired]
                if chans:
                    replace[K] = chans
            self.channels = replace

    @uricall
    def asTest(self, op, pv=None, user=None, peer=None, roles=[]):
        if not user:
            user = op.account()
            if not roles:
                roles = op.roles()
        peer = peer or op.peer().split(':')[0]
        _log.debug("asTest %s %s %s", user, roles, peer)

        if not user or not peer or not pv:
            raise RemoteError("Missing required arguments pv= user= and peer=")

        peer = socket.gethostbyname(peer)
        pvname, asg, asl = self.pvlist.compute(pv.encode('UTF-8'), peer)
        if not pvname:
            raise RemoteError("Denied")

        chan=TestChannel()
        self.acf.create(chan, asg, user, peer, asl, roles)

        return permissionsType({
            'pv':pv,
            'account':user,
            'peer':peer,
            'roles':roles,
            'asg':asg,
            'asl':asl,
            'permission':chan.perm,
        })

    @uricall
    def asDebug(self, op):
        return asDebugType(self.acf.report())

def readnproc(args, fname, fn, **kws):
    try:
        if fname:
            fullname = os.path.join(os.path.dirname(args.config), fname)
            args._all_config_files.append(fullname)
            with open(fullname, 'r') as F:
                data = F.read()
        else:
            data = ''
        return fn(data, **kws)
    except IOError as e:
        _log.error('In "%s" : %s', fname, e)
        sys.exit(1)
    except RuntimeError as e:
        _log.error('In "%s" : %s', fname, e)
        sys.exit(1)
    except ACFError as e:
        _log.error('In "%s" %s', fname, e)
        sys.exit(1)

def comment_sub(M):
    '''Replace C style comment with equivalent whitespace, includeing newlines,
       to preserve line and columns numbers in parser errors (py3 anyway)
    '''
    return re.sub(r'[^\n]', ' ', M.group(0))

def jload(raw):
    '''Parse JSON including C style comments
    '''
    return json.loads(re.sub(r'/\*.*?\*/', comment_sub, raw, flags=re.DOTALL))

def getargs():
    from argparse import ArgumentParser, ArgumentError
    P = ArgumentParser()
    P.add_argument('config', help='Config file')
    P.add_argument('--no-ban-local', action='store_true',
                    help='Skip ban of local interfaces.  Risk of GW talking to itself.')
    P.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO,
                   help='Enable basic logging with DEBUG level')
    P.add_argument('--logging', help='Use logging config from file (JSON in dictConfig format)')
    P.add_argument('--debug', action='store_true',
                   help='Enable extremely verbose low level PVA debugging')
    P.add_argument('-T', '--test-config', action='store_true',
                   help='Read and validate configuration files, then exit w/o starting a gateway.'+
                   '  Also prints the names of all configuration files read.')
    return P

class App(object):

    def __init__(self, args):
        args._all_config_files = [args.config]
        with open(args.config, 'r') as F:
            jconf = F.read()
        try:
            # we substitute comments with whitespace to keep correct line and column numbers
            # in error messages.
            jconf = jload(jconf)
            jver = jconf.get('version', 0)
            if jver not in (1,2):
                _log.error('Warning: config file version %d not in range [1, 2]\n'%jver)
        except ValueError as e:
            _log.error.write('Syntax Error in %s: %s\n'%(args.config, e.args))
            sys.exit(1)

        if not args.test_config:
            self.stats = GWStats(jconf.get('statsdb'))

        clients = {}
        statusprefix = None

        names = [jcli['name'] for jcli in jconf['clients']]
        if len(names)!=len(set(names)):
            _log.error('Duplicate client names: %s', names)
        del names

        for jcli in jconf['clients']:
            name = jcli['name']
            client_conf = {
                'EPICS_PVA_ADDR_LIST':jcli.get('addrlist',''),
                'EPICS_PVA_AUTO_ADDR_LIST':{True:'YES', False:'NO'}[jcli.get('autoaddrlist',True)],
            }
            if 'bcastport' in jcli:
                client_conf['EPICS_PVA_BROADCAST_PORT'] = str(jcli['bcastport'])
            if 'serverport' in jcli:
                client_conf['EPICS_PVA_SERVER_PORT'] = str(jcli['serverport'])

            _log.info("Client %s input config:\n%s", name, pprint.pformat(client_conf))
            if args.test_config:
                clients[name] = None
            else:
                clients[name] = _gw.Client(jcli.get('provider', u'pva'), client_conf)

        servers = self.servers = {}

        # pre-process 'servers' to expand 'interface' list
        new_servers = []
        for jsrv in jconf['servers']:
            iface = jsrv.get('interface') or ['0.0.0.0']

            if jver==1:
                # version 1 only allowed one interface.
                #  'interface':'1.2.3.4'
                # version 2 allows a list
                #  'interface':['1.2.3.4']
                if isinstance(iface, list):
                    _log.warning('Server interface list should specify JSON scheme version 2')
                else:
                    # be forgiving
                    iface = [iface]

            if len(jsrv.get('addrlist',''))>0 and len(iface)>1:
                _log.warning('Server entries for more than one interface must not specify addrlist.')
                _log.warning('Each server interface will attempt to send beacons to all destinations')
                jsrv.pop('addrlist')

            for idx, iface in enumerate(iface):
                jsrv = jsrv.copy()
                if idx!=0:
                    jsrv['name'] = '%s_%d'%(jsrv['name'], idx)
                jsrv['interface'] = iface

                new_servers.append(jsrv)

        jconf['servers'] = new_servers
        del new_servers

        names = [jsrv['name'] for jsrv in jconf['servers']]
        if len(names)!=len(set(names)):
            _log.error('Duplicate server names: %s', names)
        del names

        # various objects which shouldn't be GC'd until server shutdown,
        # but aren't otherwise accessed after startup.
        self.__lifesupport = []

        for jsrv in jconf['servers']:
            name = jsrv['name']

            providers = []

            server_conf = {
                'EPICS_PVAS_INTF_ADDR_LIST':jsrv.get('interface', '0.0.0.0'),
                'EPICS_PVAS_BEACON_ADDR_LIST':jsrv.get('addrlist', ''),
                'EPICS_PVAS_AUTO_BEACON_ADDR_LIST':{True:'YES', False:'NO'}[jsrv.get('autoaddrlist',True)],
                # ignore list not fully implemented.  (aka. never populated or used)
            }
            if 'bcastport' in jsrv:
                server_conf['EPICS_PVAS_BROADCAST_PORT'] = str(jsrv['bcastport'])
            if 'serverport' in jsrv:
                server_conf['EPICS_PVAS_SERVER_PORT'] = str(jsrv['serverport'])

            # pick client to use for ACF INP*
            aclient = jsrv.get('acf_client')
            if aclient is None:
                if len(jsrv['clients'])>1 and 'access' in jsrv:
                    _log.warning('Multiple clients and ACF is ambigious.  Add key \'acf_client\' to disambiguate')
                if len(jsrv['clients'])>0:
                    aclient = jsrv['clients'][0]

            ctxt = None
            if aclient is not None and not args.test_config:
                acli = clients[aclient]
                with acli.installAs('gwcli.'+aclient):
                    ctxt = Context('gwcli.'+aclient)

            access = readnproc(args, jsrv.get('access', ''), Engine, ctxt=ctxt)
            pvlist = readnproc(args, jsrv.get('pvlist', ''), PVList)

            if args.test_config:
                continue

            statusp = StaticProvider(u'gwsts.'+name)
            providers = [statusp]
            self.__lifesupport += [statusp]

            try:
                for client in jsrv['clients']:
                    pname = u'gws.%s.%s'%(name, client)
                    providers.append(pname)

                    client = clients[client]

                    handler = GWHandler(access, pvlist, readOnly=jconf.get('readOnly', False))
                    handler.getholdoff = jsrv.get('getholdoff')

                    if not args.test_config:
                        handler.provider = _gw.Provider(pname, client, handler) # implied installProvider()

                    self.__lifesupport += [client]
                    self.stats.handlers.append(handler)

                if 'statusprefix' in jsrv:
                    self.stats.bindto(statusp, jsrv['statusprefix'])

                    handler.asTestPV = SharedPV(nt=NTScalar('s'), initial="Only RPC supported.")
                    handler.asTestPV.rpc(handler.asTest) # TODO this is a deceptive way to assign
                    statusp.add(jsrv['statusprefix']+'asTest', handler.asTestPV)

                    handler.asDebugPV = SharedPV(nt=NTScalar('s'), initial="Only RPC supported.")
                    handler.asDebugPV.rpc(handler.asDebug) # TODO this is a deceptive way to assign
                    statusp.add(jsrv['statusprefix']+'asDebug', handler.asDebugPV)

                    # prevent client from searching for our status PVs
                    for spv in statusp.keys():
                        handler.provider.forceBan(usname=spv.encode('utf-8'))

                try:
                    server = Server(providers=providers,
                                    conf=server_conf, useenv=False)
                except RuntimeError:
                    _log.exception("Unable to create server %s", pprint.pformat(server_conf))
                    sys.exit(1)
                # we're live now...

                _log.info("Server effective config %s :\n%s", name, pprint.pformat(server.conf()))

                for spv in statusp.keys():
                    _log.info('Status PV: %s', spv)

            finally:
                [removeProvider(pname) for pname in providers[1:]]

            # keep it all from being GC'd
            servers[name] = server

        if args.test_config:
            return

        # try to prevent client -> server loops.
        # servers and clients already running, so possible race...

        if not args.no_ban_local:
            _log.info('Banning local server interfaces to prevent GW client -> GW server loops')
            for handler in self.stats.handlers:
                for server in self.servers.values():
                    server_conf = server.conf()
                    handler.provider.forceBan(host=server_conf['EPICS_PVAS_INTF_ADDR_LIST'].split(':')[0].encode('utf-8'))


    def run(self):
        try:
            while True:
                # periodic cleanup of channel cache
                _log.debug("Channel Cache sweep")
                try:
                    self.stats.sweep()
                    self.stats.update_stats()
                except:
                    _log.exception("Error during periodic sweep")

                # needs to be longer than twice the longest search interval
                self.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            [server.stop() for server in self.servers.values()]

    @staticmethod
    def sleep(dly):
        time.sleep(dly)

def main(args=None):
    args = getargs().parse_args(args)
    if args.logging is not None:
        with open(args.logging, 'r') as F:
            jconf = F.read()
        try:
            from logging.config import dictConfig
            dictConfig(jload(jconf))
        except ValueError as e:
            sys.stderr.write('%s Logging config Error: %s\n'%(args.logging, e.args))
            sys.exit(1)

    else:
        logging.basicConfig(level=args.verbose)
    if args.debug:
        set_debug(logging.DEBUG)

    app = App(args)
    if args.test_config:
        _log.info('Configuration valid')
        for fname in args._all_config_files:
            print(fname)
    else:
        app.run()

    return 0

if __name__=='__main__':
    main()
