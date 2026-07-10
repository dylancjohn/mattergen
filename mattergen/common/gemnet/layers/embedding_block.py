"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/embedding_block.py.
"""

import numpy as np
import torch

from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.common.utils.globals import MAX_ATOMIC_NUM
from neutral_layer.vocab import OS_VALUES


class IdentityEmbedding(torch.nn.Identity):
    """Embedding layer that just returns the input"""

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, with_mask_type=False):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(MAX_ATOMIC_NUM + int(with_mask_type), emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class SpeciesEmbedding(torch.nn.Module):
    """Node embeddings for (element, oxidation state) species pairs.

    ``h = e_element + gamma * e_os``, where ``gamma`` is a learned scalar
    initialised to 1.0. Element and OS have separate embedding tables so the
    model shares structure across species with the same element or the same
    oxidation state.

    Parameters
    ----------
    emb_size : int
        Embedding dimension; must match ``GemNetT.emb_size_atom``.
    with_mask_type : bool
        Whether to include a learnable MASK-token row in each embedding table.
    species_list : tuple[tuple[int, int], ...] or None
        Ordered ``(atomic_number, oxidation_state)`` pairs, matching the
        ordering in ``SpeciesVocab.species_list`` (1-based indexing).
        When ``None`` (the default), the default vocab is built via
        ``build_species_vocab()``, making this class fully Hydra-instantiable
        without an explicit species list in the config.
    """

    def __init__(
        self,
        emb_size: int,
        with_mask_type: bool = False,
        species_list: tuple[tuple[int, int], ...] | None = None,
    ):
        super().__init__()
        if species_list is None:
            from neutral_layer.vocab import build_species_vocab

            species_list = build_species_vocab().species_list
        self.emb_size = emb_size  # GemNetT reads this via getattr(atom_embedding, "emb_size")

        n_elements = MAX_ATOMIC_NUM + int(with_mask_type)  # 100 or 101
        n_os = len(OS_VALUES) + int(with_mask_type)  # 14 or 15

        self.element_embedding = torch.nn.Embedding(n_elements, emb_size)
        self.os_embedding = torch.nn.Embedding(n_os, emb_size)
        torch.nn.init.uniform_(self.element_embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        torch.nn.init.uniform_(self.os_embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        # Init to 1 so gamma * e_os starts out as a plain sum.
        self.gamma = torch.nn.Parameter(torch.ones(1))

        # Precompute 0-based lookup buffers indexed by 1-based species index.
        # Valid range: 1..num_species; MASK = num_species + 1.
        num_species = len(species_list)
        mask_index = num_species + 1
        _os_offset = -OS_VALUES[0]  # = 5: maps OS -5→0, OS 0→5, OS 8→13

        z_buf = torch.zeros(mask_index + 1, dtype=torch.long)
        os_buf = torch.zeros(mask_index + 1, dtype=torch.long)
        for idx, (z, os) in enumerate(species_list, start=1):
            z_buf[idx] = z - 1  # 0-based element index
            os_buf[idx] = os + _os_offset  # 0-based OS index
        if with_mask_type:
            z_buf[mask_index] = MAX_ATOMIC_NUM  # extra row reserved for MASK
            os_buf[mask_index] = len(OS_VALUES)  # extra row reserved for MASK

        self.register_buffer("species_to_z_idx", z_buf)
        self.register_buffer("species_to_os_idx", os_buf)

    def forward(self, species_idx: torch.Tensor) -> torch.Tensor:
        """Embed species indices.

        Parameters
        ----------
        species_idx : torch.Tensor
            1-based species indices, shape ``(N_atoms,)``.

        Returns
        -------
        h : torch.Tensor, shape ``(N_atoms, emb_size)``
        """
        z_idx = self.species_to_z_idx[species_idx]
        os_idx = self.species_to_os_idx[species_idx]
        return self.element_embedding(z_idx) + self.gamma * self.os_embedding(os_idx)


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
        activation=None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat([h_s, h_t, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st
