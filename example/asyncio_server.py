#!/usr/bin/env python
"""Demo server for demonstrating async handlers.

   $ python example/asyncio_server.py foo

"""

from __future__ import print_function

import time
import logging
import argparse
import asyncio
from p4p.server import StaticProvider, Server
from p4p.server.asyncio import SharedPV
from p4p.nt import NTScalar
import signal


DEFAULT_TIMEOUT = 1

class SomeClassWithACoroutine:
    def __init__(self):
        self.data = None

    async def coroutine(self, value: str):
        logging.info(f"Updating {self} from value `{self.data}` to `{value}`.")
        self.data = value


class AttrWHandler:
    def __init__(self, some_object_with_coro: SomeClassWithACoroutine):
        self.some_object_with_coro = some_object_with_coro

    async def put(self, pv, op):
        value = op.value()
        raw_value = value.raw.value
        logging.info(f"Received put on `{raw_value}` to `{op.name()}`.")

        await self.some_object_with_coro.coroutine(raw_value)

        pv.post(value, timestamp=time.time())
        op.done()


class AsyncProviderWrapper:
    def __init__(self, prefix: str, loop: asyncio.AbstractEventLoop):
        self.prefix = prefix
        self._loop = loop
        self._provider = StaticProvider(prefix)
        self._pvs = []

        self.setUp()

    def __del__(self):
        self.tearDown()

    @property
    def providers(self) -> tuple[StaticProvider]:
        return (self._provider,)

    async def asyncSetUp(self):
        logging.info("Async set up.")
        await self.add_pvs()

    async def add_pvs(self):
        write_pv = SharedPV(
            handler=AttrWHandler(SomeClassWithACoroutine()),
            nt=NTScalar("s"),
            initial="initial_value_1",
        )
        self._pvs.append(write_pv)
        logging.info(f"Added {self.prefix}:WRITE_PV to provider.")
        self._provider.add(f"{self.prefix}:WRITE_PV", write_pv)

        read_pv = SharedPV(
            nt=NTScalar("s"),
            initial="initial_value_2",
        )
        self._pvs.append(read_pv)
        logging.info(f"Added {self.prefix}:READ_PV to provider.")
        self._provider.add(f"{self.prefix}:READ_PV", read_pv)

    def setUp(self):
        logging.info("Sync set up.")
        self._loop.set_debug(True)
        self._loop.run_until_complete(asyncio.wait_for(self.asyncSetUp(), DEFAULT_TIMEOUT))

    def tearDown(self):
        logging.info("Sync tear down.")

class AsyncServerWrapper:
    def __init__(
        self,
        prefix: str,
    ):
        self._prefix = prefix

    def run(self):
        loop = asyncio.new_event_loop()
        provider_wrapper = AsyncProviderWrapper(self._prefix, loop)

        try:
            # `Server.forever()` is for p4p threading and shouldn't
            # be used with async.
            server = Server(provider_wrapper.providers)
            with server: 
                done = asyncio.Event()
                loop.add_signal_handler(signal.SIGINT, done.set)
                loop.add_signal_handler(signal.SIGTERM, done.set)
                loop.run_until_complete(done.wait())
        finally:
            loop.close()

def main(args: argparse.Namespace):
    AsyncServerWrapper(args.prefix).run()

def getargs() -> argparse.Namespace:
    P = argparse.ArgumentParser()
    P.add_argument('prefix', type=str)
    P.add_argument('-v','--verbose', action='store_const', default=logging.INFO, const=logging.DEBUG)
    return P.parse_args()

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=args.verbose)
    main(args)
