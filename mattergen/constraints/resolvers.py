"""resolvers.py

Custom OmegaConf resolvers for species-vocabulary configs.
"""

from functools import cache

from omegaconf import OmegaConf


@cache
def _species_vocab_size() -> int:
    """Return the number of species in the default SpeciesVocab (cached)."""
    from neutral_layer.data.vocab import build_species_vocab

    return build_species_vocab().num_species


# replace=True allows this module to be imported multiple times (e.g. in tests)
# without raising a resolver-already-registered error.
OmegaConf.register_new_resolver(
    "species_vocab_size", lambda: _species_vocab_size(), replace=True
)
