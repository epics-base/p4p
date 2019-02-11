
import logging
import time
import socket
import threading

from functools import wraps

from .nt import NTScalar, Type
from .server import Server, StaticProvider, removeProvider
from .server.thread import SharedPV, RemoteError
from . import set_debug
from . import _gw
from .asLib import Engine
from .asLib.pvlist import PVList

_log = logging.getLogger(__name__)

def getargs(*args):
    from argparse import ArgumentParser, ArgumentError
    P = ArgumentParser()
    #P.add_argument('--signore')
    P.add_argument('--sip')
    P.add_argument('--cip')
    P.add_argument('--sport', type=int, default=5076)
    P.add_argument('--cport', type=int, default=5076)
    P.add_argument('--pvlist')
    P.add_argument('--access')
    P.add_argument('--prefix')
    P.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO)
    P.add_argument('--debug', action='store_true')
    args = P.parse_args(*args)
    if not args.sip or not args.cip:
        raise ArgumentError('arguments --cip and --sip are not optional')
    return args

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
])

permissionsType = Type([
    ('pv', 's'),
    ('account', 's'),
    ('peer', 's'),
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
        self.prefix = args.prefix

        self.statusprovider = StaticProvider('gwstatus')

        self.asTestPV = SharedPV()
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
        if pvname.startswith(self.prefix):
            return self.provider.BanPV
        elif peer == self.serverep:
            _log.warn('GWS ignoring seaches from GWC: %s', peer)
            return self.provider.BanHost

        usname, _asg, _asl = self.pvlist.compute(pvname, peer.split(':',1)[0])

        if not usname:
            _log.debug("Not allowed: %s by %s", pvname, peer)
            return self.provider.BanHostPV
        else:
            ret = self.provider.testChannel(usname)
            _log.debug("allowed: %s by %s -> %s", pvname, peer, ret)
            return ret

    def makeChannel(self, op):
        _log.debug("Create %s by %s", op.name, op.peer)
        peer = op.peer.split(':',1)[0]
        usname, asg, asl = self.pvlist.compute(op.name, peer)
        if not usname:
            return None
        chan = op.create(usname)

        with self.channels_lock:
            try:
                channels = self.channels[op.peer]
            except KeyError:
                self.channels[op.peer] = channels = []
            channels.append(chan)

        try:
            self.acf.create(chan, asg, op.account, peer, asl)
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
    def asTest(self, op, pv=None, user=None, peer=None):
        # TODO: take default user and peer from op
        if not user or not peer or not pv:
            raise RemoteError("Missing required arguments pv= user= and peer=")

        peer = socket.gethostbyname(peer)
        pvname, asg, asl = self.pvlist.compute(pv, peer)
        if not pvname:
            raise RemoteError("Denied")

        chan=TestChannel()
        self.acf.create(chan, asg, user, peer, asl)

        return permissionsType({
            'pv':pv,
            'account':user,
            'peer':peer,
            'asg':asg,
            'asl':asl,
            'permission':chan.perm,
        })

def main(args):
    if args.debug:
        set_debug(logging.DEBUG)

    if args.access:
        with open(args.access, 'r') as F:
            args.access = F.read()

    if args.pvlist:
        with open(args.pvlist, 'r') as F:
            args.pvlist = F.read()

    args.access = Engine(args.access)
    args.pvlist = PVList(args.pvlist)

    handler = GWHandler(args)

    client_conf = {
        'EPICS_PVA_ADDR_LIST':args.cip,
        'EPICS_PVA_AUTO_ADDR_LIST':'NO',
        'EPICS_PVA_BROADCAST_PORT':str(args.cport),
    }
    _log.info("Client initial config: %s", client_conf)
    server_conf = {
        'EPICS_PVAS_INTF_ADDR_LIST':args.sip,
        # 'EPICS_PVAS_BEACON_ADDR_LIST': ????
        'EPICS_PVA_AUTO_ADDR_LIST':'NO',
        'EPICS_PVAS_BROADCAST_PORT':str(args.sport),
        # ignore list not fully implemented.  (aka. never populated or used)
    }

    client = _gw.installGW('gwc', client_conf, handler)
    try:
        handler.provider = client
        server= Server(providers=[handler.statusprovider, 'gwc'],
                       conf=server_conf, useenv=False)
        # we're live now...
    finally:
        removeProvider('provider')

    try:
        server_conf = server.conf()
        _log.info("Server config: %s", server_conf)

        # try to ignore myself
        handler.serverep = server_conf['EPICS_PVAS_INTF_ADDR_LIST']
        _log.debug('ignore GWS searches %s', handler.serverep)

        while True:
            # needs to be longer than twice the longest search interval
            time.sleep(60)
            # periodic cleanup of channel cache
            _log.debug("Channel Cache sweep")
            try:
                client.sweep()
                handler.sweep()
            except:
                _log.exception("Error during periodic sweep")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=args.verbose)
    main(args)
