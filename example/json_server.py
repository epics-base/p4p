#!/usr/bin/env python3

from pathlib import Path
import logging
import json

from p4p.nt import NTTable
from p4p.server import Server
from p4p.server.thread import SharedPV

logging.basicConfig()

Schema = NTTable([
    ('value', 'd'),
    ('desc',  's'),
    ('extra', 'i'),
])

initial = json.loads(Path(__file__).with_name('json_server.json').read_text())

pv = SharedPV(nt=Schema, initial=initial)

print('run: pvget example:json')

Server.forever(providers=[{
    "example:json": pv,
}])
