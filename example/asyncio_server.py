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


DEFAULT_TIMEOUT = 1

class SomeClassWithACoroutine:
    def __init__(self):
        self.data = None

    async def coroutine(self, value: str):
        logging.info(f"Updating {self} from value {self.data} to {value}.")


class AttrWHandler:
    def __init__(self, some_object_with_coro: SomeClassWithACoroutine):
        self.some_object_with_coro = some_object_with_coro

    async def put(self, pv, op):
        raw_value = op.value()
        logging.info(f"Received put on {raw_value} to {pv.name()}.")

        await self.some_object_with_coro.coroutine(raw_value)

        pv.post(raw_value, timestamp=time.time())
        op.done()


class AsyncProviderWrapper:
    def __init__(self, name: str, loop: asyncio.AbstractEventLoop):
        self.name = name
        self._loop = loop
        self._provider = StaticProvider(name)
        self._pvs = []

        self.setUp()

    async def asyncSetUp(self):
        await self.add_pvs()

    async def asyncTearDown(self): ...

    async def add_pvs(self):

        write_pv = SharedPV(
            handler=AttrWHandler(SomeClassWithACoroutine()),
            nt=NTScalar("s"),
            initial="initial_value_1",
        )
        self._pvs.append(write_pv)
        logging.info(f"Added {self.name}:WRITE_PV to provider.")
        self._provider.add(f"{self.name}:WRITE_PV", write_pv)

        read_pv = SharedPV(
            nt=NTScalar("s"),
            initial="initial_value_2",
        )
        self._pvs.append(read_pv)
        logging.info(f"Added {self.name}:READ_PV to provider.")
        self._provider.add(f"{self.name}:READ_PV", read_pv)

    def setUp(self):
        self._loop.set_debug(True)
        self._loop.run_until_complete(asyncio.wait_for(self.asyncSetUp(), DEFAULT_TIMEOUT))

    def tearDown(self):
        self._loop.run_until_complete(asyncio.wait_for(self.asyncTearDown(), DEFAULT_TIMEOUT))


class AsyncServerWrapper:
    def __init__(
        self,
        pv_prefix: str,
    ):
        self._pv_prefix = pv_prefix
        self._pvs = []

    def run(self):
        loop = asyncio.new_event_loop()
        self.provider = AsyncProviderWrapper(self._pv_prefix, loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()

    async def _run(self) -> None:
        logging.info("Running server.")
        try:
            Server.forever(providers=[self.provider._provider])
        finally:
            print("Server stopped.")


def main(args: argparse.Namespace):
    AsyncServerWrapper(args.name).run()

def getargs() -> argparse.Namespace:
    P = argparse.ArgumentParser()
    P.add_argument('prefix', type=str)
    P.add_argument('-v','--verbose', action='store_const', default=logging.INFO, const=logging.DEBUG)
    return P.parse_args()

if __name__=='__main__':
    args = getargs()
    logging.basicConfig(level=args.verbose)
    main(args)
