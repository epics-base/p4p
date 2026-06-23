import logging
logger = logging.getLogger(__name__)

from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.thread import SharedPV
from p4p.server.nthandlers import NTScalarRulesHandler

# Construct PV with control and valueAlarm structures
# and then set the values of some of those values with a post
pv = SharedPV(nt=NTScalar('d', control=True, valueAlarm=True),
              handler=NTScalarRulesHandler(),
              initial=12.0)
pv.post({'control.limitHigh': 6,
         'valueAlarm.active': True, 'valueAlarm.lowAlarmLimit': 1, 'valueAlarm.lowAlarmSeverity':2})

Server.forever(providers=[{
    'demo:pv:name':pv, # PV name only appears here
}])
