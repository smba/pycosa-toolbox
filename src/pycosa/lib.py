"""Custom exception types used across the toolbox."""


class SatisfiabilityExhaustionError(Exception):
    """Raised when the solver was asked for more solutions than exist."""

    def __init__(self, message):
        """Creates the exception with a user-facing message."""
        super().__init__(message)


class OptionExhaustionError(Exception):
    """Raised when no options are left to form a new group."""

    def __init__(self, message):
        """Creates the exception with a user-facing message."""
        super().__init__(message)
