"""
Use a handler to automatically persist values to an SQLite3 file database.
Any values persisted this way will be automatically restored when the 
program is rerun.

There are is an important caveat for this simple demo:
The persist_handler will not work as expected if anything other than the 
value of a field is changed, e.g. if a Control field was added to an NTScalar
if would not be persisted correctly. This could be resolved by correctly 
merging the pv.current() and value appropriately.
"""

import json
import random
import sqlite3
import time
from typing import Any, Tuple

from p4p import Value
from p4p.nt.scalar import NTScalar
from p4p.server import Server, ServerOperation
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV


class persist_handler(Handler):
    """
    A handler that will allow simple persistence of values and timestamps
    across retarts. It requires a post handler in order to persist values
    set within the program.
    """
    def __init__(self, pv_name: str, conn: sqlite3.Connection):
        self._conn = conn
        self._pv_name = pv_name

    def post(self, pv: SharedPV, value: Value):
        # Always update the timestamp
        if value.changed():
            now = time.time()
            value["timeStamp.secondsPastEpoch"] = int(now // 1)
            value["timeStamp.nanoseconds"] = int((now % 1) * 1e9)

        # Persist the data
        val_json = json.dumps(value.todict())

        cur = self._conn.cursor()
        cur.execute(
            """
                    INSERT INTO pvs (id, data) VALUES (?, ?)
                    ON CONFLICT(id)
                    DO UPDATE set data = ?;
                    """,
            [self._pv_name, val_json, val_json],
        )
        conn.commit()
        cur.close()

    def put(self, pv: SharedPV, op: ServerOperation):
        pv.post(op.value())  # Triggers the post() above
        op.done()


# Helper functions for restoring values on program startup
def get_initial(pv_name: str, conn: sqlite3.Connection, default=None) -> Tuple[Any, Any]:
    """
    Retrieve the initial value from the SQLite database and if there isn't
    a value then return a default value instead
    """

    cur = conn.cursor()
    res = cur.execute("SELECT data FROM pvs WHERE id=?", [pv_name])
    query_val = res.fetchone()
    cur.close()

    if query_val is not None:
        json_val = json.loads(query_val[0])
        print(f"Will restore to {pv_name} value: {json_val['value']}")
        return json_val["value"], json_val

    return default, None


def setup_pv(name: str, type: str, conn: sqlite3.Connection, default=None) -> SharedPV:
    """
    Setting up these PVs with the handler and restoring their values is
    somewhat complex!
    """
    initial, full_result = get_initial(name, conn, default)

    timestamp = None
    if full_result:
        if full_result.get("timeStamp"):
            timestamp = (
                full_result["timeStamp"]["secondsPastEpoch"]
                + full_result["timeStamp"]["nanoseconds"] / 1e9
            )

    return SharedPV(
        nt=NTScalar(type),
        handler=persist_handler(name, conn),
        initial=initial,
        timestamp=timestamp,
    )


# Create an SQLite dayabase to function as our persistence store
conn = sqlite3.connect("persist_pvs.db", check_same_thread=False)
cur = conn.cursor()
cur.execute(
    "CREATE TABLE IF NOT EXISTS pvs (id varchar(255), data json, PRIMARY KEY (id));"
)
cur.close()

duplicate_pv = setup_pv("demo:pv:int", "i", conn, default=12)
pvs = {
    "demo:pv:randint": setup_pv("demo:pv:randint", "i", conn, default=-1),
    "demo:pv:int": duplicate_pv,
    "demo:pv:float": setup_pv("demo:pv:float", "d", conn, default=9.99),
    "demo:pv:string": setup_pv("demo:pv:string", "s", conn, default="Hello!"),
    "demo:pv:alias_int": duplicate_pv,  # It works except for reporting its restore
}

print(f"Starting server with the following PVs: {pvs.keys()}")

server = None
try:
    server = Server(providers=[pvs])
    while True:
        time.sleep(1)
        pvs["demo:pv:randint"].post(random.randint(1, 1000))
except KeyboardInterrupt:
    pass
finally:
    if server:
        server.stop()
    conn.close()
