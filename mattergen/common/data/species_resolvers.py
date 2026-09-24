"""OmegaConf resolvers for species-vocabulary configs."""

from functools import cache

from omegaconf import OmegaConf


@cache
def _species_vocab_size() -> int:
    from neutral_layer.data.vocab import build_species_vocab

    return build_species_vocab().num_species


# replace=True so re-importing (e.g. in tests) does not raise
# resolver-already-registered.
OmegaConf.register_new_resolver(
    "species_vocab_size", lambda: _species_vocab_size(), replace=True
)
