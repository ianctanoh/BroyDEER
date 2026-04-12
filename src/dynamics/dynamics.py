"""
dynamics.py
Base class for dynamical systems used with DEER solvers.
"""

import equinox as eqx


class DynamicalSystem(eqx.Module):
    """
    Base class for discrete-time dynamical systems.

    Remember that when you use methods of an equinox module in a scan,
    you have to use an anonymous function.
    """

    def deer_fxn(self, state, input):
        raise NotImplementedError("Subclasses must implement deer_fxn.")

    def scan_fxn(self, state, input):
        state = self.deer_fxn(state, input)
        return state, state
