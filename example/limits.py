"""
Demonstrate a Handler which uses open() and post() to apply control.limits
to a pair of PVs.
Run the program and use
 - `pvget demo:limit_open` to verify that although the initial value was set
    to 14 the control.limitHigh means that it is set to 10.
 - `pvmonitor demo:limit_post` to see the behaviour of a PV with control limits
    set to restrict values to >=3 and <=7 as a post() sweeps it between 0 and 10.
 - `pvput demo:limit_open control.limitHigh=5` to see that changing the control
    limits is correctly supported.
"""

import time

from p4p import Value
from p4p.nt.scalar import NTScalar
from p4p.server import Server, ServerOperation
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV


class LimitsHandler(Handler):
    """
    A handler which applies the control.limits to a PV.
    """

    def _eval_limits(self, value: Value):
        """Check the value against the control.limits and adjust if necessary."""

        # Have "control.limitHigh" and "control.limitLow" been set? If they have
        # then the changed flag will be set and we can use the value. If they
        # haven't then the changed flag will not be set, they will have a default
        # value of 0, and we shouldn't use them to test the value.

        limit_high = value.get("control.limitHigh")
        if (
            limit_high is not None
            and value.changed("control.limitHigh")
            and value["value"] > limit_high
        ):
            # print(f"Value {value['value']} is above the high limit of {limit_high}, setting to {limit_high}")
            value["value"] = limit_high

        limit_low = value.get("control.limitLow")
        if (
            limit_low is not None
            and value.changed("control.limitLow")
            and value["value"] < limit_low
        ):
            # print(f"Value {value['value']} is below the low limit of {limit_low}, setting to {limit_low}")
            value["value"] = limit_low

    def open(self, value: Value):
        """Check any initial value against the limits."""
        self._eval_limits(value)

    def post(self, pv: SharedPV, value: Value):
        """Check changes to the value against the limits."""

        # The value and control limits we need to evaluate are shared between
        # the pv and the value. We need to check which fields are most current
        # (value takes precedence) and set them in the value variable
        # approppriately.
        pv_value: Value = pv.current().raw
        self._merge_field(value, pv_value, "value")
        self._merge_field(value, pv_value, "control.limitHigh")
        self._merge_field(value, pv_value, "control.limitLow")

        self._eval_limits(value)

    def _merge_field(self, value: Value, pv_value: Value, field: str):
        if not value.changed(field) and pv_value.changed(field):
            value[field] = pv_value[field]

    def put(self, pv: SharedPV, op: ServerOperation):
        """Allow puts by simply forwarding to the post and invoking the associated handler."""

        print(f"pv: {pv.current().raw}")
        print(f"op.value(): {op.value().raw}")

        pv.post(op.value())
        op.done()


def main():
    """
    Create a pvaServer with two PVs with limits.
    """

    # Construct a PV with a high limit of 10, but an initial value of 14.
    # The handler will be responsible for enforcing the limit and ensuring
    # that the value is set to 10 instead of 14.
    open_pv = SharedPV(
        nt=NTScalar("i", control=True),
        handler=LimitsHandler(),
        initial={
            "value": 14,
            "control.limitHigh": 10,
        },
    )

    # Construct a PV starting within its limit but which we will use posts
    # to step within a range of 0 to 10 every second. Observe the behaviour
    # with a pvmonitor.
    value = 5
    post_pv = SharedPV(
        nt=NTScalar("i", control=True),
        handler=LimitsHandler(),
        initial={
            "value": value,
            "control.limitLow": 3,
            "control.limitHigh": 7,
        },
    )

    pvs = {"demo:limit_open": open_pv, "demo:limit_post": post_pv}

    server = None
    try:
        server = Server(providers=[pvs])
        while True:
            time.sleep(1)
            value = value + 1
            if value > 10:
                value = 0
            # post_pv.post(value)
    except KeyboardInterrupt:
        pass
    finally:
        for pv in pvs.values():
            pv.close()
        if server:
            server.stop()


if __name__ == "__main__":
    main()
