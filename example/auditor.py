"""
In this example we setup a simple auditing mechanism that reports information
about the last channel changed (which channel, when, and who by). Since we
might need to know this information even when the program is not running we
persist this data to file, including information about when changes could
have been made.
"""

import time

from p4p.nt.scalar import NTScalar
from p4p.server import Server
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV


class Auditor(Handler):
    """Persist information to file so we can audit when the program is closed"""

    def open(self, value):
        with open("audit.log", mode="a+") as f:
            f.write(f"Auditing opened at {time.ctime()}\n")

    def close(self, pv):
        with open("audit.log", mode="a+") as f:
            value = pv.current().raw["value"]
            if value:
                f.write(f"Auditing closed at {time.ctime()}; {value}\n")
            else:
                f.write(f"Auditing closed at {time.ctime()}; no changes made\n")


class Audited(Handler):
    """Forward information about Put operations to the auditing PV"""

    def __init__(self, pv: SharedPV):
        self._audit_pv = pv

    def put(self, pv, op):
        pv.post(op.value())
        self._audit_pv.post(
            f"Channel {op.name()} last updated by {op.account()} at {time.ctime()}"
        )
        op.done()


# Setup the PV that will make the audit information available.
# Note that there is no put in its handler so it will be externally read-only
auditor_pv = SharedPV(nt=NTScalar("s"), handler=Auditor(), initial="")

# Setup some PVs that will be audited and one that won't be
# Note that the audited handler does have a put so these PVs can be changed externally
pvs = {
    "demo:pv:auditor": auditor_pv,
    "demo:pv:audited_d": SharedPV(
        nt=NTScalar("d"), handler=Audited(auditor_pv), initial=9.99
    ),
    "demo:pv:audited_i": SharedPV(
        nt=NTScalar("i"), handler=Audited(auditor_pv), initial=4
    ),
    "demo:pv:audited_s": SharedPV(
        nt=NTScalar("s"), handler=Audited(auditor_pv), initial="Testing"
    ),
    "demo:pv:unaudted_i": SharedPV(nt=NTScalar("i"), initial=-1),
}

print(pvs.keys())
try:
    Server.forever(providers=[pvs])
except KeyboardInterrupt:
    pass
finally:
    # We need to close the auditor PV manually, the server stop() won't do it for us
    auditor_pv.close()
