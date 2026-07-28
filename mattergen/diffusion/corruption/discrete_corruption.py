# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mattergen.diffusion.corruption.corruption import Corruption


class DiscreteCorruption(Corruption):
    """Shared base for categorical (non-SDE) corruption processes.

    Handles the index-offset bookkeeping common to every discrete atom-type
    corruption: MatterGen's atom/species indices are not zero-based (e.g.
    atomic numbers start at 1), so corruption math that operates on
    zero-based category indices needs to convert at the boundary.
    """

    #: Whether this corruption has a fixed number of trained noise levels
    #: (exposed via an ``N`` property) that a sampler must match exactly.
    #: True for discrete-time corruptions (D3PM); false for continuous-time
    #: corruptions (MDLM, Duo), which can be sampled with any number of steps.
    requires_fixed_num_steps: bool = False

    def __init__(self, offset: int = 0):
        # Often, the data is not zero-indexed, so we need to offset the data.
        # E.g., if we are dealing with one-based class labels, we might want to
        # offset by 1 to convert from zero-based indices to actual classes.
        self.offset = offset

    def _to_zero_based(self, x):
        """Convert from non-zero-based indices to zero-based indices."""
        return x - self.offset

    def _to_non_zero_based(self, x):
        """Convert from zero-based indices to non-zero-based indices."""
        return x + self.offset

    @property
    def T(self) -> float:
        """End time of the Corruption process."""
        return 1
