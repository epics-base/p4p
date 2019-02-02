#!/usr/bin/env python
"""Serve up an image

$ python image_server.py pv:face

Then later run in eg. ipython -pylab

  from p4p.client.thread import Context
  ctxt=Context('pva')
  imshow(ctxt.get('pv:face'))
"""

import logging
import sys

from scipy.misc import face

from p4p.nt import NTNDArray
from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV

logging.basicConfig(level=logging.INFO)

pv = SharedPV(nt=NTNDArray(), initial=face(gray=True))

provider = StaticProvider('face')
provider.add(sys.argv[1], pv)
print('serving pv:', sys.argv[1])

Server.forever(providers=[provider])
