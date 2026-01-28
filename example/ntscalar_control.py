"""
A demonstration of using a handler to apply the Control field logic for an
Normative Type Scalar (NTScalar).

There is only one PV, but it's behaviour is complex:
- try changing and checking the value, e.g.
  `python -m p4p.client.cli put demo:pv=4` and
  `python -m p4p.client.cli get demo:pv`
Initially the maximum = 11, minimum = -1, and minimum step size = 2.
Try varying the control settings, e.g.
- `python -m p4p.client.cli put demo:pv='{"value":5, "control.limitHigh":4}'`
  `python -m p4p.client.cli get demo:pv`
Remove the comments at lines 166-169 and try again.

This is also a demonstration of using the open(), put(), and post() callbacks
to implement this functionality, and particularly how it naturally partitions
the concerns of the three callback function:
- open() - logic based only on the input Value,
- post() - logic requiring comparison of cuurent and proposed Values
- put() - authorisation
"""

from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV
from p4p.wrapper import Value


class SimpleControl(Handler):
    """
    A simple handler that implements the logic for the Control field of a
    Normative Type.
    """

    def __init__(self):
        # The attentive reader may wonder why we are keeping track of state here
        # instead of relying on control.limitLow, control.limitHigh, and
        # control.minStep. There are three possible reasons a developer might
        # choose an implementation like this:
        # - As [Ref1] shows it's not straightforward to maintain state using
        #   the PV's own fields
        # - A developer may wish to have the limits apply as soon as the
        #   Channel is open. If an initial value is set then this may happen
        #   before the first post().
        # - It is possible to adapt this handler so it could be used without
        #   a Control field.
        # The disadvantage of this simple approach is that clients cannot
        # inspect the Control field values until they have been changed.
        self._min_value = None  # Minimum value allowed
        self._max_value = None  # Maximum value allowed
        self._min_step = None  # Minimum change allowed

    def open(self, value) -> bool:
        """
        This function manages all logic when we only need to consider the
        (proposed) future state of a PV
        """
        value_changed_by_limit = False

        # Check if the limitHigh has changed. If it has then we have to
        # reevaluate the existing value. Note that for this to work with a
        # post() request we have to take the actions explained at Ref1
        if value.changed("control.limitHigh"):
            self._max_value = value["control.limitHigh"]
            if value["value"] > self._max_value:
                value["value"] = self._max_value
                value_changed_by_limit = True

        if value.changed("control.limitLow"):
            self._min_value = value["control.limitLow"]
            if value["value"] < self._min_value:
                value["value"] = self._min_value
                value_changed_by_limit = True

        # This has to go in the open because it could be set in the initial value
        if value.changed("control.minStep"):
            self._min_step = value["control.minStep"]

        # If the value has changed we need to check it against the limits and
        # change it if any of the limits apply
        if value.changed("value"):
            if self._max_value and value["value"] > self._max_value:
                value["value"] = self._max_value
                value_changed_by_limit = True
            elif self._min_value and value["value"] < self._min_value:
                value["value"] = self._min_value
                value_changed_by_limit = True

        return value_changed_by_limit

    def post(self, pv: SharedPV, value: Value):
        """
        This function manages all logic when we need to know both the
        current and (proposed) future state of a PV
        """
        # [Ref1] This is where even our simple handler gets complex!
        # If the value["value"] has not been changed as part of the post()
        # operation then it will be set to a default value (i.e. 0) and
        # marked unchanged. For the logic in open() to work if the control
        # limits are changed we need to set the pv.current().raw value in
        # this case.
        if not value.changed("value"):
            value["value"] = pv.current().raw["value"]
            value.mark("value", False)

        # Apply the control limits before the check for minimum change because:
        # - the self._min_step may be updated
        # - the value["value"] may be altered by the limits
        value_changed_by_limit = self.open(value)

        # If the value["value"] wasn't changed by the put()/post() but was
        # changed by the limits then we don't check the min_step but
        # immediately return
        if value_changed_by_limit:
            return

        if (
            self._min_step
            and abs(pv.current().raw["value"] - value["value"]) < self._min_step
        ):
            value.mark("value", False)

    def put(self, pv, op):
        """
        In most cases the combination of a put() and post() means that the
        put() is solely concerned with issues of authorisation.
        """
        # Demo authorisation.
        # Only Alice may remotely change the Control limits
        # Bob is forbidden from changing anything on this Channel
        # Everyone else may change the value but not the Control limits
        errmsg = None
        if op.account() == "Alice":
            pass
        elif op.account() == "Bob":
            op.done(error="Bob is forbidden to make changes!")
            return
        else:
            if op.value().raw.changed("control"):
                errmsg = f"Unauthorised attempt to set Control by {op.account()}"
                op.value().raw.mark("control", False)

        # Because we have not set use_handler_post=False in the post this
        # will automatically trigger evaluation of the post rule and thus
        # the application of
        pv.post(op.value())
        op.done(error=errmsg)


# Construct a PV with Control fields and use a handler to apply the Normative
# Type logic. Note that the Control logic is correctly applied even to the
# initial value, based on the limits set in the rest of the initial value.
pv = SharedPV(
    nt=NTScalar("d", control=True),
    handler=SimpleControl(),
    initial={
        "value": 12.0,
        "control.limitHigh": 11,
        "control.limitLow": -1,
        "control.minStep": 2,
    },  # Immediately limited to 11 due to handler
)


# Override the put in the handler so that we can perform puts for testing
# @pv.on_put
# def handle(pv, op):
#     pv.post(op.value())  # just store and update subscribers
#     op.done()


pvs = {
    "demo:pv": pv,
}
print("PVs: ", pvs)
Server.forever(providers=[pvs])
