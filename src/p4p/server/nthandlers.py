""" Handler for NTScalar (so far) """

import logging
import operator
import time

from collections import OrderedDict
from enum import Enum
from __future__ import annotations
from typing import Callable

from p4p import Value
from p4p.server import ServerOperation
from p4p.server.thread import Handler, SharedPV

logger = logging.getLogger(__name__)


class BaseRulesHandler(Handler):
    """
    Base class for rules that includes rules common to all PV types.
    """
    class RulesFlow(Enum):
        """What to do after a rule has been evaluated"""

        CONTINUE = 1  # Continue rules processing
        TERMINATE = 2  # Do not process more rules but we're good to here
        ABORT = 3  # Stop rules processing and abort put

    def __init__(self) -> None:
        # TODO: Removed timestamp so the logic for this will need to replicated somewhere else

        super().__init__()
        self._name = None

        self._init_rules : OrderedDict[
            Callable[[dict, Value], Value]
        ] = OrderedDict()

        self._put_rules : OrderedDict[
            Callable[[SharedPV, ServerOperation], self.RulesFlow]
        ] = OrderedDict()


    def OnFirstConnect(self, pv : SharedPV) -> None:
        """
        This method is called by the pvrecipe after the pv has been created
        """
        #Evaluate the timestamp last
        self._init_rules.move_to_end("timestamp")

        for post_init_rule_name, post_init_rule in self._init_rules.items():
            logger.debug('Processing post init rule %s', post_init_rule_name)
            value = post_init_rule(pv.current().raw, pv.current().raw)
            if value:
                pv.post(value=value)

    def put(self, pv: SharedPV, op: ServerOperation) -> None:
        """Put that applies a set of rules"""
        self._name = op.name()
        logger.debug("Processing attempt to change PV %s by %s (member of %s) at %s",
                     op.name(), op.account(), op.roles(), op.peer())

        # oldpvstate : Value = pv.current().raw
        newpvstate: Value = op.value().raw

        logger.debug(
            "Processing changes to the following fields: %r (value = %s)",
            newpvstate.changedSet(),
            newpvstate["value"],
        )

        if not self._apply_rules(pv, op):
            return

        logger.info(
            "Making the following changes to %s: %r",
            self._name,
            newpvstate.changedSet(),
        )
        pv.post(op.value())  # just store and update subscribers

        op.done()
        logger.info("Processed change to PV %s by %s (member of %s) at %s",
                    op.name(), op.account(), op.roles(), op.peer())

    def _apply_rules(self, pv: SharedPV, op: ServerOperation) -> bool:
        """
        Apply the rules, usually when a put operation is attempted
        """
        for rule_name, put_rule in self._put_rules.items():
            logger.debug('Applying rule %s', rule_name)
            rule_flow = put_rule(pv, op)

            match (rule_flow):
                case self.RulesFlow.CONTINUE:
                    pass
                case self.RulesFlow.ABORT:
                    logger.debug("Rule %s triggered handler abort", rule_name)
                    op.done()
                    return False
                case self.RulesFlow.TERMINATE:
                    logger.debug("Rule %s triggered handler terminate", rule_name)
                    break
                case None:
                    logger.warning(
                        "Rule %s did not return rule flow. Defaulting to "
                        "CONTINUE, but this behaviour may change in "
                        "future.",
                        rule_name,
                    )
                case _:
                    logger.critical("Rule %s returned unhandled return type", rule_name)
                    raise TypeError(
                        f"Rule {rule_name} returned unhandled return type {type(rule_flow)}"
                    )

        return True

    def set_read_only(self):
        """
        Make this PV read only.
        """
        self._put_rules["read_only"] = (
            lambda new, old: BaseRulesHandler.RulesFlow.ABORT
        )
        self._put_rules.move_to_end("read_only", last=False)

    def _timestamp_rule(self, _, op: ServerOperation) -> RulesFlow:
        """Handle updating the timestamps"""

        # Note that timestamps are not automatically handled so we may need to set them ourselves
        newpvstate: Value = op.value().raw
        self.evaluate_timestamp(_, newpvstate)

        return self.RulesFlow.CONTINUE

    def evaluate_timestamp(self, _ : dict, newpvstate : Value) -> Value:
        """ Update the timeStamp of a PV """
        if newpvstate.changed("timeStamp"):
            logger.debug("Using timeStamp from put operation")
        else:
            logger.debug("Generating timeStamp from time.time()")
            sec, nsec = time_in_seconds_and_nanoseconds(time.time())
            newpvstate["timeStamp.secondsPastEpoch"] = sec
            newpvstate["timeStamp.nanoseconds"] = nsec

        return newpvstate
