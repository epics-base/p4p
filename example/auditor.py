"""
In this example we setup a simple auditing mechanism that reports information
about the last audited PV changed (which PV, when, and who by) to an auditor
channel. The record of changes is written to the file `audit.log` file.

NOTE: This code is an example only. This design is not suitable for non-demo
      use.

The two Handlers below demonstrate many of the methods available to a handler.
The Audited handler records only information about (external) `put()`
operations, not (internal) `post()` operations. The Auditor Handler uses
`open()` to record the start of logging, `post()` to record value changes,
and `close()` to record when auditing finished and write its own value at that
time.

Run this script and then make a change to one of its PVs, e.g.
`python -m p4p.client.cli put demo:pv:audited_d=8.8`. Then check the results of
the change, e.g. `python -m p4p.client.cli get demo:pv:auditor`. You should see
the name of the PV that was changed, the identity of the user that made the
change, and the time the change was made.

After making changes you can inspect the log of changes in the "audit.log".
Use of open() and close() means that it's also possible to check when auditing
started and stopped.
"""

import time

from p4p import Value
from p4p.nt.scalar import NTScalar
from p4p.server import Server
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV


class Auditor(Handler):
    """Persist information to file so we can audit when the program is closed"""

    def open(self, value: Value, **_kws):
        """Record the time the auditing PV was opened."""
        with open("audit.log", mode="a+", encoding="utf8") as f:
            f.write(f"Auditing opened at {time.ctime()}\n")

    def post(self, pv, value: Value, **_kws):
        """Record the time a change was to the auditing PV, and the change made."""
        with open("audit.log", mode="a+", encoding="utf8") as f:
            f.write(f"Auditing updated at {time.ctime()}; {value['value']}\n")

    def close(self, pv: Value):
        """Record the time the auditing PV was closed."""
        with open("audit.log", mode="a+", encoding="utf8") as f:
            value = pv.current().raw["value"]
            if value:
                f.write(f"Auditing closed at {time.ctime()}; {value}\n")
            else:
                f.write(f"Auditing closed at {time.ctime()}; no changes made\n")


class Audited(Handler):
    """Forward information about Put operations to the auditing PV."""

    def __init__(self, pv: SharedPV):
        self._audit_pv = pv

    def put(self, pv, op):
        """Each time a Put operation is made we forward some information to the auditing PV."""
        pv.post(op.value())
        self._audit_pv.post(
            f"Channel {op.name()} last updated by {op.account()} at {time.ctime()}"
        )
        op.done()


def main():
    """
    Create a set of text PVs. The audited channels forward information about the user that
    made a change and their new values to an auditor channel.
    """

    # Setup some PVs that will be audited and one (`demo:pv:unaudted_i`) that won't be
    # Note that the audited handler does have a put so it can also be changed externally
    auditor_pv = SharedPV(nt=NTScalar("s"), handler=Auditor(), initial="")

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


if __name__ == "__main__":
    main()
