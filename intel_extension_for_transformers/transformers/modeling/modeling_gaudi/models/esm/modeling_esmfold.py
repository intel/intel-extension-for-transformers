# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
from transformers.models.esm.modeling_esmfold import EsmForProteinFoldingOutput, categorical_lddt
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)
from transformers.utils import (
    ContextManagers,
)


def gaudi_esmfolding_trunk_forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
    """
    Inputs:
        seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
        x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

    Output:
        predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object

    Copied from EsmFoldingTrunk.forward:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esmfold.py
    The change is:
    - Add extra mark_step in trunk_iter for each block.
    """

    device = seq_feats.device
    s_s_0 = seq_feats
    s_z_0 = pair_feats

    if no_recycles is None:
        no_recycles = self.config.max_recycles
    else:
        if no_recycles < 0:
            raise ValueError("Number of recycles must not be negative.")
        no_recycles += 1  # First 'recycle' is just the standard forward pass through the model.

    def trunk_iter(s, z, residx, mask):
        z = z + self.pairwise_positional_embedding(residx, mask=mask)

        for block in self.blocks:
            s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            if s.device.type == "hpu":
                import habana_frameworks.torch.core as htcore

                htcore.mark_step()
        return s, z

    s_s = s_s_0
    s_z = s_z_0
    recycle_s = torch.zeros_like(s_s)
    recycle_z = torch.zeros_like(s_z)
    recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

    for recycle_idx in range(no_recycles):
        with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
            # === Recycling ===
            recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
            recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
            recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

            s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

            # === Structure module ===
            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask.float(),
            )

            recycle_s = s_s
            recycle_z = s_z
            # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
            recycle_bins = self.distogram(
                structure["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.recycle_bins,
            )

    structure["s_s"] = s_s
    structure["s_z"] = s_z

    return structure


def gaudi_esm_for_protein_folding_forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    masking_pattern: Optional[torch.Tensor] = None,
    num_recycles: Optional[int] = None,
) -> EsmForProteinFoldingOutput:
    r"""
    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, EsmForProteinFolding

    >>> model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  # A tiny random peptide
    >>> outputs = model(**inputs)
    >>> folded_positions = outputs.positions
    ```

    Copied from EsmForProteinFolding.forward:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esmfold.py
    The change is:
    - rewrite (softmax().unsqueeze() @ esm_s).squeeze() with equivalent but less dims algorithm on HPU.

    """
    cfg = self.config.esmfold_config

    aa = input_ids  # B x L
    B = aa.shape[0]
    L = aa.shape[1]
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(aa, device=device)
    if position_ids is None:
        position_ids = torch.arange(L, device=device).expand_as(input_ids)

    # === ESM ===
    esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

    if masking_pattern is not None:
        masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
    else:
        masked_aa = aa
        mlm_targets = None

    # We get sequence and pair representations from whatever version of ESM /
    # configuration we are using. The sequence representation esm_s is always
    # present. The pair embedding esm_z may be present depending on the
    # configuration of the model. If esm_z is not used by the model then it
    # is returned as None here.
    esm_s = self.compute_language_model_representations(esmaa)

    # Convert esm_s and esm_z, if present, to the precision used by the trunk and
    # the structure module. These tensors may be a lower precision if, for example,
    # we're running the language model in fp16 precision.
    esm_s = esm_s.to(self.esm_s_combine.dtype)

    if cfg.esm_ablate_sequence:
        esm_s = esm_s * 0

    esm_s = esm_s.detach()

    # === preprocessing ===
    if esm_s.device.type == "hpu":
        dims = esm_s.shape
        esm_s = esm_s.reshape(-1, dims[-2], dims[-1])  # combine first 2 dims
        esm_s = self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s
        esm_s = esm_s.reshape(dims[0], dims[1], esm_s.shape[-2], esm_s.shape[-1])  # split back 1st dim
        esm_s = esm_s.squeeze(2)
    else:
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = self.esm_s_mlp(esm_s)

    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

    if self.config.esmfold_config.embed_aa:
        s_s_0 += self.embedding(masked_aa)

    structure: dict = self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
    # Documenting what we expect:
    structure = {
        k: v
        for k, v in structure.items()
        if k
        in [
            "s_z",
            "s_s",
            "frames",
            "sidechain_frames",
            "unnormalized_angles",
            "angles",
            "positions",
            "states",
        ]
    }

    # Add BERT mask for the loss to use, if available.
    if mlm_targets:
        structure["mlm_targets"] = mlm_targets

    disto_logits = self.distogram_head(structure["s_z"])
    disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
    structure["distogram_logits"] = disto_logits

    lm_logits = self.lm_head(structure["s_s"])
    structure["lm_logits"] = lm_logits

    structure["aatype"] = aa
    make_atom14_masks(structure)
    # Of course, this doesn't respect the true mask because it doesn't know about it...
    # We're not going to properly mask change of index tensors:
    #    "residx_atom14_to_atom37",
    #    "residx_atom37_to_atom14",
    for k in [
        "atom14_atom_exists",
        "atom37_atom_exists",
    ]:
        structure[k] *= attention_mask.unsqueeze(-1)
    structure["residue_index"] = position_ids

    lddt_head = self.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, self.lddt_bins)
    structure["lddt_head"] = lddt_head
    plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
    structure["plddt"] = plddt

    ptm_logits = self.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

    return EsmForProteinFoldingOutput(**structure)


def gaudi_rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer to avoid AMP downcasting.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates

    Copied from rot_vec_mul:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/openfold_utils/rigid_utils.py
    The change is:
    - Using matmul when possible on HPU to get better performance.
    """
    # Do matmal on HPU directly when possible to get better performance.
    if r.device.type == "hpu":
        if t.dim() > 5:
            pass
        elif t.dim() == 5:
            # Combine shape[2] and shape[3] on HPU
            shape_t = t.shape
            shape_r = r.shape
            t = t.reshape(shape_t[0], shape_t[1], shape_t[2] * shape_t[3], shape_t[4])
            r = r.reshape(shape_r[0], shape_r[1], shape_r[2] * shape_r[3], shape_r[4], shape_r[5])
            t = t.unsqueeze(-2)
            r = r.transpose(-2, -1)
            out = t @ r
            shape_out = out.shape
            out = out.reshape(
                shape_out[0],
                shape_out[1],
                max(shape_r[2], shape_t[2]),
                max(shape_r[3], shape_t[3]),
                shape_out[3],
                shape_out[4],
            )
            out = out.squeeze(-2)
            return out
        else:
            t = t.unsqueeze(-2)
            r = r.transpose(-2, -1)
            out = t @ r
            out = out.squeeze(-2)
            return out

    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def gaudi_rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication of two rotation matrix tensors. Written out by hand to avoid AMP downcasting.

    Args:
        a: [*, 3, 3] left multiplicand
        b: [*, 3, 3] right multiplicand
    Returns:
        The product ab

    Copied from rot_matmul:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/openfold_utils/rigid_utils.py
    The change is:
    - Using matmul when possible on HPU to get better performance.
    """

    # Do matmal on HPU directly when possible to get better performance.
    if a.device.type == "hpu":
        if a.shape == b.shape or a.dim() < 5:
            out = a @ b
            return out
        elif a.dim() == 5 and a.shape[2] == 1:
            # HPU does not handle dim==5 with below broadcast correctly.
            # a.shape = torch.Size([1, 512, 1, 3, 3]), b.shape = torch.Size([1, 512, 8, 3, 3])
            a = a.permute(0, 1, 2, 4, 3)
            b = b.permute(0, 1, 2, 4, 3)
            out = b @ a
            out = out.permute(0, 1, 2, 4, 3)
            return out
        else:
            pass

    def row_mul(i: int) -> torch.Tensor:
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )
