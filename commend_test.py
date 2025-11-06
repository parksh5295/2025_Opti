"""Temporary file for change review demo."""


def demo_function(x: int, y: int) -> int:
    """Return the sum of two numbers."""

    return x + y


class DemoAccumulator:
    """Simple stateful accumulator for demonstration purposes."""

    def __init__(self, *, initial: int = 0) -> None:
        """Initialise the running total.

        Parameters
        ----------
        initial:
            Optional starting value for the accumulator. Must be non-negative.
        """

        if initial < 0:
            raise ValueError("initial must be non-negative")
        self.total = initial

    def add(self, value: int) -> int:
        """Add a value to the running total and return the new total."""

        self.total += value
        return self.total

    def subtract(self, value: int) -> int:
        """Subtract a value and return the updated total."""

        self.total -= value
        return self.total

    def reset(self) -> None:
        """Reset the total back to zero."""

        self.total = 0

