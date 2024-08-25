from p4p.nt import NTScalar
from p4p.server import Server
from p4p.server.raw import Handler
from p4p.server.thread import SharedPV
from p4p.wrapper import Type, Value


class SimpleControl(Handler):
    """
    A simple handler that implements the logic for the Control field of a
    Normative Type.
    """

    def __init__(self, min_value=None, max_value=None, min_step=None):
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
        self._min_value = min_value  # Minimum value allowed
        self._max_value = max_value  # Maximum value allowed
        self._min_step = min_step  # Minimum change allowed

    def open(self, value):
        """
        This function manages all logic when we only need to consider the
        (proposed) future state of a PV
        """
        # Check if the limitHigh has changed. If it has then we have to reevaluate
        # the existing value. Note that for this to work with a post request we
        # have to take the actions explained at Ref1
        if value.changed("control.limitHigh"):
            self._max_value = value["control.limitHigh"]
            if value["value"] > self._max_value:
                value["value"] = self._max_value

        if value.changed("control.limitLow"):
            self._min_value = value["control.limitLow"]
            if value["value"] < self._min_value:
                value["value"] = self._min_value

        # If the value has changed we need to check it against the limits and
        # change it if any of the limits apply
        if value.changed("value"):
            if self._max_value and value["value"] > self._max_value:
                value["value"] = self._max_value
            if self._min_value and value["value"] < self._min_value:
                value["value"] = self._min_value

    def post(self, pv, value):
        """
        This function manages all logic when we need to know both the
        current and (proposed) future state of a PV
        """
        # If the minStep has changed update this instance's minStemp value
        if value.changed("control.minStep"):
            self._min_change = value["control.minStep"]

        # [Ref1] This is where even our simple handler gets complex!
        # If the value["value"] has not been changed as part of the post()
        # operation then it will be set to a default value (i.e. 0) and
        # marked unchanged.
        current_value = pv.current().raw
        value_changed = True  # TODO: Explain this
        if not value.changed("value"):
            value["value"] = current_value["value"]
            value.mark("value", False)
            value_changed = False

        # Apply the control limits before the check for minimum change as the
        # value may be altered by the limits.
        self.open(value)
        if not value_changed and value.changed("value"):
            return

        if abs(current_value["value"] - value["value"]) < self._min_step:
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
                errmsg = (
                    f"Unauthorised attempt to set Control by {op.account()}"
                )
                op.value().raw.mark("control", False)

        # Because we have not set use_handler_post=False in the post this
        # will automatically trigger evaluation of the post rule and thus
        # the application of
        pv.post(op.value())
        op.done(error=errmsg)


# Construct PV with control and structures
# and then set the values of some of those values with a post
pv = SharedPV(
    nt=NTScalar("d", control=True),
    handler=SimpleControl(-1, 11, 2),
    initial=12.0,  # Immediately limited to 11 due to handler on live above
)
pv.post(
    {
        "control.limitHigh": 6,  # Value now limited to 6
    }
)


@pv.on_put
def handle(pv, op):
    pv.post(op.value())  # just store and update subscribers
    op.done()


print("demo:pv:name: ", pv)
Server.forever(
    providers=[
        {
            "demo:pv:name": pv,
        }
    ]
)
