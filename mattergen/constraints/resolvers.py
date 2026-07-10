"""resolvers.py

Custom OmegaConf resolvers for species-vocabulary configs.

Registers the ``species_vocab_size`` resolver so YAML configs can reference
the vocabulary size without hardcoding it:

    dim: ${eval:'${species_vocab_size:} + 1'}

The resolver is lazy: SMACT is not queried until a config value backed by
``${species_vocab_size:}`` is actually accessed.

Modules
-------
_species_vocab_size
    Cached helper that returns ``SpeciesVocab.num_species`` for the default vocab.
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
OmegaConf.register_new_resolver("species_vocab_size", lambda: _species_vocab_size(), replace=True)
