import sys
import logging
import time
import socket
import threading
import platform

from functools import wraps

from .nt import NTScalar, Type
from .server import Server, StaticProvider, removeProvider
from .server.thread import SharedPV, RemoteError
from . import set_debug
from . import _gw
from .asLib import Engine
from .asLib.pvlist import PVList

if sys.version_info >= (3, 0):
    unicode = str

_log = logging.getLogger(__name__)

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

statsType = Type([
    ('ccacheSize', NTScalar.buildType('L')),
    ('mcacheSize', NTScalar.buildType('L')),
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
    ])),
], id='epics:p2p/Permission:1.0')

class GWHandler(object):
    def __init__(self, args):
        self.acf, self.pvlist = args.access, args.pvlist
        self.serverep = None
        self.channels_lock = threading.Lock()
        self.channels = {}
        self.prefix = args.prefix and args.prefix.encode('UTF-8')

        self.statusprovider = StaticProvider('gwstatus')

        self.asTestPV = SharedPV(nt=NTScalar('s'), initial="Only RPC supported.")
        self.asTestPV.rpc(self.asTest) # TODO this is a deceptive way to assign
        if args.prefix:
            self.statusprovider.add(args.prefix+'asTest', self.asTestPV)

        self.clientsPV = SharedPV(nt=NTScalar('as'), initial=[])
        if args.prefix:
            self.statusprovider.add(args.prefix+'clients', self.clientsPV)

        self.cachePV = SharedPV(nt=NTScalar('as'), initial=[])
        if args.prefix:
            self.statusprovider.add(args.prefix+'cache', self.cachePV)

        self.statsPV = SharedPV(initial=statsType())
        if args.prefix:
            self.statusprovider.add(args.prefix+'stats', self.statsPV)

        if args.prefix:
            for name in self.statusprovider.keys():
                _log.info("status PV: %s", name)

    def testChannel(self, pvname, peer):
        if self.prefix and pvname.startswith(self.prefix):
            _log.debug("GWS ignores own status")
            return self.provider.BanPV
        elif peer == self.serverep:
            _log.warn('GWS ignoring seaches from GWC: %s', peer)
            return self.provider.BanHost

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
            self.acf.create(chan, asg, op.account, peer, asl, op.roles)
        except:
            # create() should fail secure.  So allow this client to
            # connect R/O.  We already acknowledged the search, so
            # if we fail here the client will go into a reset loop.
            _log.exception("Default restrictive for %s from %s", op.name, op.peer)
        return chan

    def sweep(self):
        replace = {}
        with self.channels_lock:
            for K, chans in self.channels.items():
                chans = [chan for chan in chans if not chan.expired]
                if chans:
                    replace[K] = chans
            self.channels = replace

        self.clientsPV.post(self.channels.keys())
        self.cachePV.post(self.provider.cachePeek())
        self.statsPV.post(statsType(self.provider.stats()))

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

class IFInfo(object):
    def __init__(self, ep):
        addr, _sep, port = ep.partition(':')
        self.port = int(port or '5076')

        info = _gw.IFInfo.current(socket.AF_INET, socket.SOCK_DGRAM, unicode(addr))
        for iface in info:
            if iface['addr']==addr:
                self.__dict__.update(iface)
                return
        raise ValueError("Not local interface %s"%addr)

    @property
    def addr_list(self):
        ret = [self.addr]
        if hasattr(self, 'bcast'):
            ret.append(self.bcast)
        elif self.loopback and self.addr=='127.0.0.1' and platform.system()=='Linux':
            # On Linux, the loopback interface is not advertised as supporting broadcast (IFF_BROADCAST)
            # but actually does.  We are assuming here the conventional 127.0.0.1/8 configuration.
            ret.append('127.255.255.255')
        return ' '.join(ret)

    @staticmethod
    def show():
        _log.info("Local network interfaces")
        for iface in _gw.IFInfo.current(socket.AF_INET, socket.SOCK_DGRAM):
            _log.info("%s", iface)

class App(object):
    @staticmethod
    def getargs(*args):
        from argparse import ArgumentParser, ArgumentError
        P = ArgumentParser()
        #P.add_argument('--signore')
        P.add_argument('--server', help='Server interface address, with optional port (default 5076)')
        P.add_argument('--cip', type=lambda v:set(v.split()), default=set(),
                       help='Client address list, with optional ports (defaults set by --cport)')
        P.add_argument('--cport', help='Client default port', type=int, default=5076)
        P.add_argument('--pvlist', help='Optional PV list file.  Default allows all')
        P.add_argument('--access', help='Optional ACF file.  Default allows all')
        P.add_argument('--prefix', help='Prefix for status PVs')
        P.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO)
        P.add_argument('--debug', action='store_true')
        args = P.parse_args(*args)
        if not args.server or len(args.cip)==0:
            raise ArgumentError('arguments --cip and --server are not optional')
        return args

    def __init__(self, args):
        if args.access:
            with open(args.access, 'r') as F:
                args.access = F.read()

        if args.pvlist:
            with open(args.pvlist, 'r') as F:
                args.pvlist = F.read()

        args.access = Engine(args.access)
        args.pvlist = PVList(args.pvlist)

        IFInfo.show()

        srv_iface = IFInfo(args.server)

        local_bcast = set([iface['bcast'] for iface in _gw.IFInfo.current(socket.AF_INET, socket.SOCK_DGRAM) if 'bcast' in iface])

        if not args.cip.intersection(local_bcast):
            _log.warn('Client address list includes no local interface broadcast addresses.')
            _log.warn('These are: %s', ', '.join(local_bcast))

        self.handler = GWHandler(args)

        client_conf = {
            'EPICS_PVA_ADDR_LIST':' '.join(args.cip),
            'EPICS_PVA_AUTO_ADDR_LIST':'NO',
            'EPICS_PVA_BROADCAST_PORT':str(args.cport),
        }
        _log.info("Client initial config: %s", client_conf)
        server_conf = {
            'EPICS_PVAS_INTF_ADDR_LIST':srv_iface.addr,
            'EPICS_PVAS_BEACON_ADDR_LIST':srv_iface.addr_list,
            'EPICS_PVA_AUTO_ADDR_LIST':'NO',
            'EPICS_PVAS_BROADCAST_PORT':str(srv_iface.port),
            # ignore list not fully implemented.  (aka. never populated or used)
        }

        self.client = _gw.installGW(u'gwc', client_conf, self.handler)
        try:
            self.handler.provider = self.client
            self.server= Server(providers=[self.handler.statusprovider, 'gwc'],
                        conf=server_conf, useenv=False)
            # we're live now...
        finally:
            removeProvider(u'gwc')

        server_conf = self.server.conf()
        _log.info("Server config: %s", server_conf)
        # try to ignore myself
        self.handler.serverep = server_conf['EPICS_PVAS_INTF_ADDR_LIST']
        _log.debug('ignore GWS searches %s', self.handler.serverep)

    def run(self):
        try:
            while True:
                # needs to be longer than twice the longest search interval
                self.sleep(60)
                # periodic cleanup of channel cache
                _log.debug("Channel Cache sweep")
                try:
                    self.client.sweep()
                    self.handler.sweep()
                except:
                    _log.exception("Error during periodic sweep")
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()

    @staticmethod
    def sleep(dly):
        time.sleep(dly)

def main(args=None):
    args = App.getargs(args)
    logging.basicConfig(level=args.verbose)
    if args.debug:
        set_debug(logging.DEBUG)
    App(args).run()

if __name__=='__main__':
    main()
