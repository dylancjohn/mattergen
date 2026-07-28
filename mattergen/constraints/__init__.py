"""constraints

Charge-vocabulary glue for charge-neutral generation (``build_charge_of``) that has
no non-constrained equivalent elsewhere in the codebase. The diffusion-family-specific
constrained losses/samplers (D3PM, MDLM, Duo) live alongside their unconstrained
counterparts in ``mattergen.diffusion.<family>``; the species dataset/datamodule/
resolvers live in ``mattergen.common.data``.
"""
