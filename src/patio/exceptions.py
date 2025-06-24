class PatioData(RuntimeError):  # noqa: N818
    pass


class NoNonZeroCEMS(PatioData):
    pass


class NoCEMS(PatioData):
    pass


class NoREData(PatioData):
    pass


class NoNonZero923(PatioData):
    pass


class NoFiniteNonZeroCapAdj(PatioData):
    pass


class NoMatchingPlantsAndCEMS(PatioData):
    pass


class CEMSExtensionError(PatioData):
    pass


class NoEligibleCleanRepowering(PatioData):
    pass


class ScenarioError(RuntimeError):
    """Duplicate or broken scenario."""

    pass


class NoMaxRE(RuntimeError):  # noqa: N818
    """Duplicate or broken scenario."""

    pass
