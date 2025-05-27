"""
Use a handler to automatically persist values to an SQLite3 file database.
Any values persisted this way will be automatically restored when the
program is rerun. The details of users (account name and IP address) are
recorded for puts.

Try monitoring the PV `demo:pv:optime` then quit, wait, and restart the
program while continuing to monitor the PV. Compare with the value of
`demo:pv:uptime` which resets on each program start. Try setting the value of
demo:pv:optime while continuing to monitor it. It is recommended to
inspect the persisted file, e.g. `sqlite3 persist_pvs.db "select * from pvs"`.

There is an important caveat for this simple demo:
The `PersistHandler` will not work as expected if anything other than the
value of a field is changed, e.g. if a Control field was added to an NTScalar
if would not be persisted correctly. This could be resolved by correctly
merging the pv.current().raw and value.raw appropriately in the post().
"""

import json
import sqlite3
import time

from p4p import Value
from p4p.nt.scalar import NTScalar
from p4p.server import Server, ServerOperation
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV


class PersistHandler(Handler):
    """
    A handler that will allow simple persistence of values and timestamps
    across retarts. It requires a post handler in order to persist values
    set within the program.
    """

    def __init__(self, pv_name: str, conn: sqlite3.Connection, open_restore=True):
        self._conn = conn
        self._pv_name = pv_name
        self._open_restore = open_restore

    def open(self, value, **kws):
        # If there is a value already in the database we always use that
        # instead of the supplied initial value, unless the
        # handler_open_restore flag indicates otherwise.
        if not self._open_restore:
            return

        # We could, in theory, re-apply authentication here if we queried for
        # that information and then did something with it!
        res = self._conn.execute("SELECT data FROM pvs WHERE id=?", [self._pv_name])
        query_val = res.fetchone()

        if query_val is not None:
            json_val = json.loads(query_val[0])
            print(f"Will restore to {self._pv_name} value: {json_val['value']}")

            # Override initial value
            value["value"] = json_val["value"]

            value["timeStamp.secondsPastEpoch"] = json_val["timeStamp"][
                "secondsPastEpoch"
            ]
            value["timeStamp.nanoseconds"] = json_val["timeStamp"]["nanoseconds"]
        else:
            # We are using an initial value so persist it
            self._upsert(value)

    def post(
        self,
        pv: SharedPV,
        value: Value,
    ):
        self._update_timestamp(value)

        self._upsert(
            value,
        )

    def put(self, pv: SharedPV, op: ServerOperation):
        # The post does all the real work, we just add info only available
        # from the ServerOperation
        self._update_timestamp(op.value())

        self._upsert(
            op.value(), op.account(), op.peer()
        )

        op.done()

    def _update_timestamp(self, value) -> None:
        if not value.changed("timeStamp") or (
            value["timeStamp.nanoseconds"] == value["timeStamp.nanoseconds"] == 0
        ):
            now = time.time()
            value["timeStamp.secondsPastEpoch"] = now // 1
            value["timeStamp.nanoseconds"] = int((now % 1) * 1e9)

    def _upsert(self, value, account=None, peer=None) -> None:
        # Persist the data; turn into JSON and write it to the DB
        val_json = json.dumps(value.todict())

        # Use UPSERT: https://sqlite.org/lang_upsert.html
        conn.execute(
            """
            INSERT INTO pvs (id, data, account, peer) 
                        VALUES (:name, :json_data, :account, :peer)
            ON CONFLICT(id)
            DO UPDATE SET data = :json_data, account = :account, peer = :peer;
            """,
            {
                "name": self._pv_name,
                "json_data": val_json,
                "account": account,
                "peer": peer,
            },
        )
        conn.commit()


# Create an SQLite dayabase to function as our persistence store
conn = sqlite3.connect("persist_pvs.db", check_same_thread=False)
#conn.execute("DROP TABLE IF EXISTS pvs")
conn.execute(
    "CREATE TABLE IF NOT EXISTS pvs (id VARCHAR(255), data JSON, account VARCHAR(30), peer VARCHAR(55), PRIMARY KEY (id));"
)  # IPv6 addresses can be long and will contain port number as well!

duplicate_pv = SharedPV(
    nt=NTScalar("i"), handler=PersistHandler("demo:pv:int", conn), initial=12
)
pvs = {
    "demo:pv:optime": SharedPV(
        nt=NTScalar("i"),
        handler=PersistHandler("demo:pv:optime", conn),
        initial=0,
    ),  # Operational time; total time running
    "demo:pv:uptime": SharedPV(
        nt=NTScalar("i"),
        handler=PersistHandler("demo:pv:uptime", conn, open_restore=False),
        timestamp=time.time(),
        initial=0,
    ),  # Uptime since most recent (re)start
    "demo:pv:int": duplicate_pv,
    "demo:pv:float": SharedPV(
        nt=NTScalar("d"),
        handler=PersistHandler("demo:pv:float", conn),
        initial=9.99,
    ),
    "demo:pv:string": SharedPV(
        nt=NTScalar("s"),
        handler=PersistHandler("demo:pv:string", conn),
        initial="Hello!",
    ),
    "demo:pv:alias_int": duplicate_pv,  # It works except for reporting its restore
}


# Make the uptime PV readonly; maybe we want to be able to update optime
# after major system upgrades?
uptime_pv = pvs["demo:pv:uptime"]


@uptime_pv.put
def read_only(pv: SharedPV, op: ServerOperation):
    op.done(error="Read-only")
    return


print(f"Starting server with the following PVs: {pvs}")

server = None
try:
    server = Server(providers=[pvs])
    while True:
        # Every second increment the values of uptime and optime
        time.sleep(1)
        increment_value = pvs["demo:pv:uptime"].current().raw["value"] + 1
        pvs["demo:pv:uptime"].post(increment_value)
        increment_value = pvs["demo:pv:optime"].current().raw["value"] + 1
        pvs["demo:pv:optime"].post(increment_value)
except KeyboardInterrupt:
    pass
finally:
    if server:
        server.stop()
    conn.close()
