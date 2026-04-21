"""
A very minimal example showing how to map a python Enum to an NTEnum.

in another terminal use
 `python -m p4p.client.cli put demo:pv:name=Watermelon` and
 `python -m p4p.client.cli put demo:pv:name=10`
  and see how you end up with the same result for both commands.
"""

from enum import Enum

from p4p.nt import NTEnum
from p4p.server import Server
from p4p.server.thread import SharedPV


class Fruit(Enum):
    Apple = 0
    Pear = 1
    Orange = 2
    Lemon = 3
    Mango = 4
    Lime = 5
    Melon = 6
    Grapefruit = 7
    Strawberry = 8
    Blackberry = 9
    Watermelon = 10
    Yuzu = 11
    Banana = 12
    Pineapple = 13
    Peach = 14
    Grape = 15


pv = SharedPV(
    nt=NTEnum(),
    initial={"choices": [x.name for x in Fruit], "index": Fruit.Strawberry.value},
)


@pv.put
def handle(pv, op):
    print(f"My favourite fruit is {op.value()}")
    pv.post(op.value())  # just store and update subscribers
    op.done()


Server.forever(
    providers=[
        {
            "demo:pv:name": pv,  # PV name only appears here
        }
    ]
)  # runs until KeyboardInterrupt
